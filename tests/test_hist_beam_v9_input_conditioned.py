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
    prediction_marginal_kl_loss,
    widened_target_prior,
)
from kd_sensing.engine.hist_beam_loso_summary import row_eligibility  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import (  # noqa: E402
    collapse_diagnostics_payload,
    histogram_kl,
    write_collapse_diagnostics,
)
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402


def _v9_model(**kwargs) -> HistBeamFusionNet:
    defaults = {
        "modalities": ["gps"],
        "feature_size": 8,
        "d_model": 16,
        "num_classes": 12,
        "num_pred": 2,
        "group_size": 4,
        "variant": "v9_input_conditioned_target_adaptation",
        "num_heads": 4,
        "num_layers": 1,
        "v8": {"beta_prior": 0.5},
        "v9": {"beta_prior_max": 1.0, "prototype_type": "beam", "prototype_tau": 0.2},
    }
    defaults.update(kwargs)
    return HistBeamFusionNet(**defaults)


def test_v9_forward_combines_target_prior_and_prototype_without_source_logits():
    model = _v9_model(v9={"beta_prior_max": 1.0, "prototype_type": "beam", "prototype_tau": 0.2, "eta_prototype": 0.7})
    prototypes = torch.randn(12, 16)
    counts = torch.ones(12, dtype=torch.long)
    output = model(gps_batch=torch.randn(3, 2, 3), target_prototypes=prototypes, target_prototype_counts=counts)

    beta = output["hist_beam"]["v8_beta_prior"]
    expected = output["target_logits"] + beta * output["target_prior_bias"] + 0.7 * output["prototype_logits"]

    assert output["hist_beam"]["v9_input_conditioned_target_adaptation"] is True
    assert output["hist_beam"]["source_logits_in_final"] is False
    assert output["prototype_logits"].shape == (3, 2, 12)
    assert torch.allclose(output["logits_final"], expected, atol=1e-5)
    assert not torch.allclose(output["logits_final"], output["source_logits"])
    assert 0.0 <= beta <= 1.0


def test_v9_fixed_beta_freeze_and_prior_dropout_train_only():
    model = _v9_model(v9={"learnable_beta_prior": False, "beta_prior_max": 0.5, "prior_dropout": 1.0, "use_prototype_logits": False})
    with torch.no_grad():
        model.target_prior_bias.fill_(2.0)
    strategy = apply_hist_beam_adaptation_strategy(model, "v9_target_head_only")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    model.train()
    dropped = model(gps_batch=torch.randn(2, 2, 3))
    model.eval()
    kept = model(gps_batch=torch.randn(2, 2, 3))

    assert strategy["v9_target_head_only_freeze_strategy"] is True
    assert not any(name.startswith("beta_prior_raw") for name in trainable)
    assert dropped["hist_beam"]["v9_prior_dropout_active"] is True
    assert kept["hist_beam"]["v9_prior_dropout_active"] is False
    assert not torch.allclose(dropped["logits_final"], kept["logits_final"])


def test_v9_beam_and_sector_target_prototypes_from_support_features():
    model = _v9_model()
    features = torch.randn(5, 16)
    labels = torch.tensor([0, 1, 1, 6, 11])
    beam_meta = model.set_target_prototypes_from_features(features, labels, prototype_type="beam")
    beam_out = model(gps_batch=torch.randn(2, 2, 3))

    sector_model = _v9_model(v9={"prototype_type": "sector", "sector_size": 3, "prototype_tau": 0.2})
    sector_meta = sector_model.set_target_prototypes_from_features(features, labels, prototype_type="sector", sector_size=3)
    sector_out = sector_model(gps_batch=torch.randn(2, 2, 3))

    assert beam_meta["target_prototype_support_count"] == 5
    assert beam_out["hist_beam"]["prototype_logits_available"] is True
    assert sector_meta["target_prototype_available_count"] >= 2
    assert sector_out["prototype_logits"].shape == (2, 2, 12)
    assert sector_out["hist_beam"]["v9_sector_mapping"] == "floor_division_shared_score_to_member_beams"


def test_v9_widened_prior_kl_and_collapse_diagnostics_are_stable(tmp_path: Path):
    labels = torch.tensor([[1, 2], [2, 3], [3, 3]])
    logits = torch.randn(3, 2, 6)
    support = torch.tensor([0, 0, 3, 1, 0, 0], dtype=torch.float32)
    widened = widened_target_prior(support, sigma=2.0, temperature=1.5)
    loss = prediction_marginal_kl_loss(logits, widened)
    path = write_collapse_diagnostics(
        tmp_path / "collapse_diagnostics.json",
        labels,
        logits,
        num_classes=6,
        support_prior=widened,
        target_logits=logits,
        target_prior_bias=torch.log(widened).view(1, 1, -1).expand(3, 2, -1),
        prototype_logits=torch.zeros_like(logits),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert torch.isfinite(loss)
    assert widened.argmax().item() == 2
    assert histogram_kl([0, 1, 0], [0, 0, 1]) > 0
    assert payload["kl_pred_support"] >= 0
    assert payload["evaluation_only_target_test_label"] is True
    assert "target_logits_only" in payload["branches"]
    assert payload["per_true_beam_confusion"]


def test_v9_loss_adds_non_kd_anti_collapse_diagnostics():
    model = _v9_model(v9={"use_widened_prior_marginal_kl": True})
    output = model(gps_batch=torch.randn(3, 2, 3))
    labels = torch.tensor([[1, 2], [2, 3], [3, 4]])
    loss = compute_hist_beam_loss(
        output,
        labels,
        cfg={
            "hist_beam": {
                "variant": "v9_input_conditioned_target_adaptation",
                "num_classes": 12,
                "group_size": 4,
                "v9": {
                    "use_widened_prior_marginal_kl": True,
                    "support_prior": [1.0 / 12] * 12,
                    "widened_prior_sigma": 2.0,
                    "widened_prior_temperature": 1.5,
                },
                "loss_weights": {"v9_widened_prior_marginal_kl": 0.2},
            }
        },
    )

    assert loss.total.isfinite()
    assert loss.diagnostics["hist/v9/widened_prior_marginal_kl_enabled"] == 1.0
    assert loss.diagnostics["hist/v9/anti_collapse_loss"] >= 0.0


def test_v9_eligibility_allows_evaluation_only_target_test_and_excludes_oracle_usage():
    eligible = row_eligibility(
        {"run_status": "completed"},
        {
            "main_conclusion_eligible": True,
            "used_target_oracle_fields": [],
            "target_test_label_usage": "evaluation_only",
        },
        {"target_test_label_usage": "evaluation_only"},
    )
    excluded = row_eligibility(
        {"run_status": "completed", "used_target_path_label_for_training": True},
        {
            "main_conclusion_eligible": False,
            "eligibility_reasons": ["target_path_label_supervision"],
            "used_target_oracle_fields": ["target_path_label"],
            "target_oracle_usage_stage": {"target_path_label": "target_adaptation_training_loss"},
        },
        {},
    )

    assert eligible["main_conclusion_eligible"] is True
    assert excluded["main_conclusion_eligible"] is False
    assert "target_oracle_fields_used" in excluded["eligibility_reasons"]
