import copy

import pytest
import torch

from kd_sensing.eval.pcpf import (
    _control_fusion_from_cached_evidence,
    build_stage2_gate_report,
    fit_train_confidence_p90,
    summarize_pcpf_matrix,
)
from tools.eval_pcpf import _require_control_comparability


MODALITIES = ["image", "radar", "gps", "lidar"]
PATTERNS = [
    "full",
    "missing_image",
    "missing_radar",
    "missing_gps",
    "missing_lidar",
    "image_only",
    "radar_only",
    "gps_only",
    "lidar_only",
    "missing_image_radar",
    "missing_image_gps",
    "missing_image_lidar",
    "missing_radar_gps",
    "missing_radar_lidar",
    "missing_gps_lidar",
]


def _records() -> dict:
    samples = 120
    labels = torch.arange(samples).remainder(64)
    base = torch.linspace(0.01, 0.99, samples).unsqueeze(1)
    risk = base + torch.tensor([[0.0, 0.01, 0.02, 0.03]])
    probabilities = torch.full((samples, 4, 64), 0.1 / 63)
    probabilities.scatter_(2, labels.view(-1, 1, 1).expand(-1, 4, 1), 0.9)
    probabilities[::7, :, 0] = 0.95
    probabilities[::7] /= probabilities[::7].sum(dim=-1, keepdim=True)
    available = torch.ones(samples, 4, dtype=torch.bool)
    weights = torch.softmax(-risk, dim=-1)
    fused = (weights.unsqueeze(-1) * probabilities).sum(dim=1)
    weather_names = ("sunny", "rainy", "foggy")
    weather = [weather_names[index % 3] for index in range(samples)]
    domains = [f"{weather[index]}/scene{index % 15}" for index in range(samples)]
    patterns = [PATTERNS[index % len(PATTERNS)] for index in range(samples)]
    mask_groups = [
        "full" if name == "full" else "single" if name.endswith("_only") else "drop1" if name.count("_") == 1 else "drop2"
        for name in patterns
    ]
    static = torch.full_like(weights, 0.25)
    return {
        "labels": labels,
        "raw_risk": risk,
        "target_risk": risk.clone(),
        "available": available,
        "fusion_weights": weights,
        "unimodal_probabilities": probabilities,
        "fused_probability": fused,
        "risk_components": torch.stack([risk, risk * 2, risk * 3, risk * 4], dim=-1),
        "weather": weather,
        "domain": domains,
        "pattern": patterns,
        "mask_group": mask_groups,
        "modalities": MODALITIES,
        "replacement_probability": {
            "uniform": probabilities.mean(dim=1),
            "static_prior": probabilities.mean(dim=1),
            "pcpf_analytic": fused,
        },
        "replacement_weights": {
            "uniform": static,
            "static_prior": static,
            "pcpf_analytic": weights,
        },
        "modality_temperatures": torch.ones(4),
        "static_capability": torch.ones(4),
        "fusion_tau": 1.0,
        "expert_fingerprint": "a" * 64,
        "training_stage": "stage2_risk",
        "bounded_evaluation": False,
    }


def _gate() -> dict:
    return {
        "overall_spearman_min": 0.2,
        "minimum_positive_modalities": 3,
        "require_each_weather_positive": True,
        "upper_lower_gap_min": 0.0,
        "minimum_risk_std": 1e-4,
    }


def test_stage2_gate_passes_registered_observability_contract() -> None:
    records = _records()
    confidence, count = fit_train_confidence_p90(records)

    report = build_stage2_gate_report(
        records,
        _gate(),
        train_confidence_p90=confidence,
        train_confidence_count=count,
        stage2_checkpoint_sha256="b" * 64,
        bounded_evaluation=False,
    )

    assert report["stage2_gate_passed"] is True
    assert report["failure_reasons"] == []
    assert report["overall"]["spearman"] > 0.99
    assert len(report["domains"]) == 15
    assert set(report["mask_groups"]) == {"full", "drop1", "drop2", "single"}


def test_bounded_gate_is_never_promotion_eligible() -> None:
    records = _records()
    confidence, count = fit_train_confidence_p90(records)

    report = build_stage2_gate_report(
        records,
        _gate(),
        train_confidence_p90=confidence,
        train_confidence_count=count,
        stage2_checkpoint_sha256="b" * 64,
        bounded_evaluation=True,
    )

    assert report["stage2_gate_passed"] is False
    assert "bounded_evaluation_not_gate_eligible" in report["failure_reasons"]


def test_matrix_reuses_same_unimodal_probabilities_for_replacements() -> None:
    records = _records()
    confidence, _ = fit_train_confidence_p90(records)
    provenance = {
        "checkpoint": {"path": "/tmp/stage3_best.pth", "sha256": "c" * 64, "role": "validation_best"},
        "data_protocol": {"protocol_fingerprint": "d" * 64, "validation_role": "inner_validation"},
        "normalization": {"source_split": "inner_train", "risk_component_std": [0.01, 0.2, 0.3, 0.4]},
    }

    report = summarize_pcpf_matrix(records, train_confidence_p90=confidence, provenance=provenance)

    assert set(report["overall"]["replacement_metrics"]) == {"uniform", "static_prior", "pcpf_analytic"}
    assert report["overall"]["weight_diagnostics"]["missing_weight_max"] == 0.0
    assert report["overall"]["weight_diagnostics"]["weight_row_sum_max_error"] < 1e-6
    assert report["direct_router_status"] == "not_supplied"
    assert report["provenance"] == provenance
    assert set(report["pattern_aggregates"]) == {"single", "all14"}
    expected_single = sum(
        report["patterns"][name]["replacement_metrics"]["pcpf_analytic"]["top1"]
        for name in PATTERNS
        if name.endswith("_only")
    ) / 4
    assert report["pattern_aggregates"]["single"]["pcpf_analytic"]["top1"]["macro"] == pytest.approx(
        expected_single
    )


def test_matrix_rejects_missing_provenance() -> None:
    records = _records()
    confidence, _ = fit_train_confidence_p90(records)

    with pytest.raises(ValueError, match="data_protocol, normalization"):
        summarize_pcpf_matrix(records, train_confidence_p90=confidence, provenance={"checkpoint": {}})


def test_control_fusion_uses_cached_main_evidence_and_control_calibration() -> None:
    class Control:
        fusion_mode = "cuaf_local_adaptation"

        def __init__(self) -> None:
            self.calls = 0

        def _fuse(self, logits, probabilities, risk, components, available):
            self.calls += 1
            assert torch.equal(probabilities, diagnostics["unimodal_probabilities"])
            assert torch.equal(risk, diagnostics["raw_risk"])
            assert torch.equal(components, diagnostics["risk_components"])
            calibrated = torch.softmax(logits / 2.0, dim=-1) * available.unsqueeze(-1)
            weights = torch.softmax(torch.zeros_like(risk).masked_fill(~available, -torch.inf), dim=-1)
            return weights, calibrated, torch.ones(4), self.fusion_mode

    diagnostics = {
        "unimodal_logits": torch.randn(2, 4, 64),
        "unimodal_probabilities": torch.softmax(torch.randn(2, 4, 64), dim=-1),
        "raw_risk": torch.rand(2, 4),
        "risk_components": torch.rand(2, 4, 4),
    }
    available = torch.tensor([[True, True, True, True], [True, False, False, False]])
    control = Control()

    weights, calibrated = _control_fusion_from_cached_evidence(control, diagnostics, available)

    assert control.calls == 1
    assert torch.equal(weights[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(calibrated, torch.softmax(diagnostics["unimodal_logits"] / 2.0, dim=-1) * available.unsqueeze(-1))


def test_control_comparability_rejects_seed_or_budget_mismatch() -> None:
    reference = {
        "experiment": {"seed": 1},
        "data_protocol": {
            "protocol_id": "clean",
            "protocol_fingerprint": "a" * 64,
            "train_role": "inner_train",
            "validation_role": "inner_validation",
        },
        "temporal_missing": {"seed": 0, "mode": "balanced"},
        "data": {"dataloader": {"train_batch_size": 32, "validation_batch_size": 32}},
        "training": {
            "epochs": 10,
            "max_epochs": 10,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "checkpoint_selection": "best_validation_loss",
            "initialization_checkpoint": {
                "sha256": "b" * 64,
                "role": "validation_best",
                "expected_source_training_stage": "stage2_risk",
            },
        },
        "scheduler": {"type": "none"},
    }
    control = copy.deepcopy(reference)

    assert _require_control_comparability(reference, control, mode="uniform")["seed"] == 1

    control["experiment"]["seed"] = 2
    control["training"]["epochs"] = 5
    with pytest.raises(ValueError, match="seed, training_budget"):
        _require_control_comparability(reference, control, mode="uniform")
