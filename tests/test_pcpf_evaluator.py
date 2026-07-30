import copy
import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd_sensing.eval.pcpf import (
    _control_fusion_from_cached_evidence,
    build_stage2_gate_report,
    fit_train_confidence_p90,
    resolve_pcpf_missing_patterns,
    summarize_pcpf_matrix,
)
from kd_sensing.models.pcpf_temporal_risk import analytic_fusion_weights
from tools.eval_pcpf import (
    _evaluation_normalization_metadata,
    _load_reusable_evidence,
    _load_historical_reference_summary,
    _require_control_comparability,
    _sequential,
)


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


def test_evaluation_prefers_run_normalization_binding(tmp_path) -> None:
    checkpoint_artifacts = {"gps_scaler": "run-local.npz", "metadata": {"fit_split": "train"}}
    checkpoint = tmp_path / "best.pth"
    torch.save({"normalization_artifacts": checkpoint_artifacts}, checkpoint)
    cfg = {
        "data": {"normalization_artifacts": {"gps_scaler": "upstream.npz"}},
        "runtime": {"normalization_artifacts": checkpoint_artifacts},
    }

    assert _evaluation_normalization_metadata(cfg, checkpoint) == {
        "normalization_artifacts": checkpoint_artifacts
    }


def test_sequential_evaluation_loader_keeps_workers_and_order() -> None:
    loader = DataLoader(TensorDataset(torch.arange(7)), batch_size=3, shuffle=True, num_workers=1)

    sequential = _sequential(loader)

    assert sequential.num_workers == 1
    assert torch.cat([batch[0] for batch in sequential]).tolist() == list(range(7))


def test_reusable_evidence_validates_identity_and_adds_derived_fields(tmp_path) -> None:
    class Model:
        training_stage = "stage1_expert"
        modalities = ("image", "radar", "gps", "lidar", "csi")

        @staticmethod
        def _expert_fingerprint() -> str:
            return "a" * 64

    path = tmp_path / "records.pt"
    torch.save(
        {
            "training_stage": "stage1_expert",
            "modalities": list(Model.modalities),
            "expert_fingerprint": "a" * 64,
            "bounded_evaluation": False,
            "pattern": ["full", "full"],
            "trained_controls": [],
            "labels": torch.tensor([1, 2]),
            "unimodal_probabilities": torch.rand(2, 5, 64),
            "available": torch.ones(2, 5, dtype=torch.bool),
        },
        path,
    )

    records = _load_reusable_evidence(
        path,
        model=Model(),
        expected_patterns={"full"},
        samples_per_pattern=2,
        expected_controls=set(),
    )

    assert records["unimodal_confidence"].shape == (2, 5)
    assert records["unimodal_correct"].dtype == torch.bool


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


def test_sparse_csi_matrix_uses_exact_r_d_contracts() -> None:
    records = _five_modality_records()
    train_records = _five_modality_records(pattern_limit="full")
    confidence, _ = fit_train_confidence_p90(train_records)
    provenance = {
        "checkpoint": {"path": "/tmp/stage3_best.pth", "sha256": "c" * 64, "role": "validation_best"},
        "data_protocol": {"protocol_fingerprint": "d" * 64, "validation_role": "inner_validation"},
        "normalization": {"source_split": "inner_train", "risk_component_std": [0.1] * 4},
    }

    report = summarize_pcpf_matrix(
        records,
        train_confidence_p90=confidence,
        provenance=provenance,
        diagnostics_config={
            "bootstrap": {"seed": 1, "resamples": 8, "confidence": 0.95},
            "historical_reference_summary": {"status": "verified_test_reference"},
        },
    )

    assert len(report["patterns"]) == 31
    assert report["pattern_aggregates"]["all30"]["pcpf_analytic"]["top1"]["pattern_count"] == 30
    assert report["pattern_aggregates"]["csi_present_with_sensing"]["pcpf_analytic"]["top1"]["pattern_count"] == 15
    assert report["pattern_aggregates"]["csi_absent_legacy15"]["pcpf_analytic"]["top1"]["pattern_count"] == 15
    assert len(report["pattern_domain"]) == 31 * 3
    assert set(report["expert_input_diagnostics"]["prototype_distance"]) == set(records["modalities"])
    assert report["R0_R7"]["R1_five_modality_checkpoint_csi_masked"]["status"] == (
        "evaluated_on_all_legacy_nonempty_sensing_masks"
    )
    assert set(report["R0_R7"]) == {
        "R0_four_modality_pcpf",
        "R1_five_modality_checkpoint_csi_masked",
        "R2_five_modality_uniform",
        "R3_five_modality_static_prior",
        "R4_five_modality_direct_router",
        "R5_five_modality_cuaf_local_adaptation",
        "R6_five_modality_pcpf_analytic",
        "R7_joint_checkpoint_csi_only",
    }
    methods = report["mechanism_diagnostics"]["dynamicity_test"]["methods"]
    assert set(methods) == {
        "D0_original_sample_risk",
        "D1_domain_mask_shuffled_risk",
        "D2_domain_mask_mean_risk",
        "D3_static_prior",
    }


def test_historical_r0_loader_requires_hashed_unbounded_inner_report(tmp_path) -> None:
    records = _records()
    confidence, _ = fit_train_confidence_p90(records)
    report = summarize_pcpf_matrix(
        records,
        train_confidence_p90=confidence,
        provenance={"checkpoint": {"sha256": "c" * 64}, "data_protocol": {}, "normalization": {}},
    )
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    summary = _load_historical_reference_summary({"path": str(path), "sha256": digest})

    assert summary["status"] == "historical_reference_reused_without_recomputation"
    assert summary["report"]["sha256"] == digest
    assert summary["overall15"]["count"] == 120


def _five_modality_records(*, pattern_limit: str | None = None) -> dict:
    modalities = ["image", "radar", "gps", "lidar", "csi"]
    patterns = resolve_pcpf_missing_patterns(["all_nonempty"], modalities)
    if pattern_limit is not None:
        patterns = {pattern_limit: patterns[pattern_limit]}
    samples = 6
    labels_base = torch.arange(samples).remainder(64)
    generator = torch.Generator().manual_seed(7)
    base_probability = torch.rand(samples, 5, 64, generator=generator)
    base_probability.scatter_add_(
        2,
        labels_base.view(-1, 1, 1).expand(-1, 5, 1),
        torch.full((samples, 5, 1), 4.0),
    )
    base_probability /= base_probability.sum(dim=-1, keepdim=True)
    base_risk = torch.rand(samples, 5, generator=generator) + torch.arange(5).float().view(1, -1) * 0.03
    capability = torch.tensor([1.0, 0.9, 1.1, 0.8, 1.2])
    labels = []
    availability = []
    probability = []
    risk = []
    names = []
    weather = []
    domains = []
    sample_ids = []
    group_ids = []
    for sample in range(samples):
        for pattern, mask in patterns.items():
            available = torch.tensor(mask, dtype=torch.bool)
            labels.append(labels_base[sample])
            availability.append(available)
            probability.append(base_probability[sample] * available.unsqueeze(-1))
            risk.append(base_risk[sample] * available)
            names.append(pattern)
            weather.append(("sunny", "rainy", "foggy")[sample % 3])
            domains.append(f"{weather[-1]}/scene0")
            sample_ids.append(f"sample-{sample}")
            group_ids.append(f"trajectory-{sample // 2}")
    labels_tensor = torch.stack(labels).long()
    available_tensor = torch.stack(availability)
    probability_tensor = torch.stack(probability)
    risk_tensor = torch.stack(risk)
    uniform = available_tensor.float() / available_tensor.sum(dim=1, keepdim=True)
    static = capability.view(1, -1) * available_tensor
    static /= static.sum(dim=1, keepdim=True)
    dynamic = analytic_fusion_weights(
        risk=risk_tensor,
        available=available_tensor,
        static_capability=capability,
        tau=1.0,
    )
    fused = (dynamic.unsqueeze(-1) * probability_tensor).sum(dim=1)
    uniform_probability = (uniform.unsqueeze(-1) * probability_tensor).sum(dim=1)
    static_probability = (static.unsqueeze(-1) * probability_tensor).sum(dim=1)
    components = torch.stack((risk_tensor, risk_tensor * 2, risk_tensor * 0.5, risk_tensor * 1.5), dim=-1)
    return {
        "labels": labels_tensor,
        "unimodal_logits": probability_tensor.clamp_min(1e-12).log(),
        "raw_risk": risk_tensor,
        "target_risk": risk_tensor.clone(),
        "available": available_tensor,
        "fusion_weights": dynamic,
        "unimodal_probabilities": probability_tensor,
        "calibrated_unimodal_probabilities": probability_tensor,
        "fused_probability": fused,
        "risk_components": components,
        "normalized_risk_components": components,
        "csi_log_rms": torch.zeros(labels_tensor.numel(), 5),
        "csi_valid_ratio": torch.ones(labels_tensor.numel(), 5),
        "csi_quality_confidence": torch.linspace(0.2, 1.0, labels_tensor.numel()).unsqueeze(1).expand(-1, 5),
        "csi_snr_available": torch.zeros(labels_tensor.numel(), 5, dtype=torch.bool),
        "weather": weather,
        "domain": domains,
        "pattern": names,
        "mask_group": ["full" if name == "full" else "other" for name in names],
        "sample_id": sample_ids,
        "group_id": group_ids,
        "modalities": modalities,
        "replacement_probability": {
            "uniform": uniform_probability,
            "static_prior": static_probability,
            "pcpf_analytic": fused,
            "direct_router_control": uniform_probability,
            "cuaf_local_adaptation": static_probability,
        },
        "replacement_weights": {
            "uniform": uniform,
            "static_prior": static,
            "pcpf_analytic": dynamic,
            "direct_router_control": uniform,
            "cuaf_local_adaptation": static,
        },
        "modality_temperatures": torch.ones(5),
        "static_capability": capability,
        "fusion_tau": 1.0,
        "risk_coefficients": torch.ones(4),
        "risk_component_mean": torch.zeros(4),
        "risk_component_std": torch.ones(4),
        "trained_controls": ["direct_router_control", "cuaf_local_adaptation"],
        "replacement_parameters": {},
        "expert_fingerprint": "e" * 64,
        "training_stage": "stage3_fusion",
        "bounded_evaluation": False,
    }


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
