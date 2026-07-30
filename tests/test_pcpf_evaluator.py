import torch

from kd_sensing.eval.pcpf import build_stage2_gate_report, fit_train_confidence_p90, summarize_pcpf_matrix


MODALITIES = ["image", "radar", "gps", "lidar"]


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
    group_names = ("full", "drop1", "drop2", "single")
    weather = [weather_names[index % 3] for index in range(samples)]
    domains = [f"{weather[index]}/scene{index % 15}" for index in range(samples)]
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
        "pattern": [f"pattern{index % 15}" for index in range(samples)],
        "mask_group": [group_names[index % 4] for index in range(samples)],
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

    report = summarize_pcpf_matrix(records, train_confidence_p90=confidence)

    assert set(report["overall"]["replacement_metrics"]) == {"uniform", "static_prior", "pcpf_analytic"}
    assert report["overall"]["weight_diagnostics"]["missing_weight_max"] == 0.0
    assert report["overall"]["weight_diagnostics"]["weight_row_sum_max_error"] < 1e-6
    assert report["direct_router_status"] == "not_supplied"
