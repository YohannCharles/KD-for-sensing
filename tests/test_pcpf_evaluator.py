import copy
import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd_sensing.config import load_config
from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.eval.pcpf import (
    _control_fusion_from_cached_evidence,
    _r0_r7_summary,
    build_stage2_gate_report,
    fit_train_confidence_p90,
    resolve_pcpf_missing_patterns,
    summarize_pcpf_matrix,
)
from kd_sensing.models.pcpf_temporal_risk import analytic_fusion_weights
from kd_sensing.utils.checkpoint import publish_checkpoint
from tools.eval_pcpf import (
    _comparison_budget,
    _evaluation_normalization_metadata,
    _load_evaluation_checkpoint,
    _load_reusable_evidence,
    _load_trajectory_r0_reference,
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


def _sha256_lines(values: list[str], *, sort: bool = False) -> str:
    selected = sorted(values) if sort else values
    return hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()


def _identity_binding(records: dict) -> dict[str, object]:
    indices = [index for index, pattern in enumerate(records["pattern"]) if pattern == "full"]
    protocol_ids = [records["protocol_sample_id"][index] for index in indices]
    return {
        "sample_count": len(indices),
        "sample_id_sha256": _sha256_lines(protocol_ids, sort=True),
    }


def _protocol(records: dict, *, trajectory: bool = False) -> dict[str, object]:
    binding = _identity_binding(records)
    return {
        "mode": TRAJECTORY_PROTOCOL_MODE,
        "protocol_id": TRAJECTORY_PROTOCOL_MODE,
        "protocol_fingerprint": "d" * 64,
        "audit_id": "mmw_id_stratified_block_audit_v1",
        "audit_sha256": "a" * 64,
        "split_seed": 0,
        "train_role": "train",
        "validation_role": "validation",
        "train_sample_count": 100,
        "validation_sample_count": binding["sample_count"],
        "train_sample_id_hash": "c" * 64,
        "validation_sample_id_hash": binding["sample_id_sha256"],
        "outer_test_enabled": False,
    }


def _comparison_budget_descriptor() -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "pcpf_trajectory_r0_r7_budget_v1",
        "stage_epochs": {"stage1_expert": 40, "stage2_risk": 20, "stage3_fusion": 10},
        "stage_learning_rates": {"stage1_expert": 0.0005, "stage2_risk": 0.0005, "stage3_fusion": 0.0001},
        "train_batch_size": 64,
        "validation_batch_size": 64,
        "scheduler": "none",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _topology(*, formal: bool = False) -> dict[str, object]:
    return {
        "id": "ula_dft_phase_cycle_v1" if formal else "cyclic_index_v1",
        "descriptor_sha256": "1" * 64 if formal else "",
        "audit_path": "/tmp/topology.json" if formal else "",
        "audit_sha256": "2" * 64 if formal else "",
        "formal_r0_r7_eligible": formal,
    }


def _provenance(records: dict, *, trajectory: bool = False, comparison_budget: bool = False) -> dict:
    result = {
        "checkpoint": {"path": "/tmp/stage3_best.pth", "sha256": "c" * 64, "role": "validation_best"},
        "data_protocol": _protocol(records, trajectory=trajectory),
        "normalization": {
            "source_split": "train",
            "risk_component_std": [0.01, 0.2, 0.3, 0.4],
        },
        "experiment_seed": 1,
        "prototype_topology": _topology(formal=trajectory and comparison_budget),
        "validation_identity_binding": _identity_binding(records),
    }
    if comparison_budget:
        result["comparison_budget"] = _comparison_budget_descriptor()
    return result


def _trajectory_r0_summary(records: dict, provenance: dict) -> dict[str, object]:
    indices = [index for index, pattern in enumerate(records["pattern"]) if pattern == "full"]
    sample_ids = [records["sample_id"][index] for index in indices]
    protocol_ids = [records["protocol_sample_id"][index] for index in indices]
    protocol = provenance["data_protocol"]
    return {
        "status": "verified_same_protocol_trajectory_reference",
        "comparison_contract": {
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "protocol_audit_id": protocol["audit_id"],
            "protocol_audit_sha256": protocol["audit_sha256"],
            "train_role": protocol["train_role"],
            "validation_role": protocol["validation_role"],
            "experiment_seed": provenance["experiment_seed"],
            "validation_sample_count": len(indices),
            "validation_ordered_sample_id_sha256": _sha256_lines(sample_ids),
            "validation_protocol_sample_id_sha256": _sha256_lines(protocol_ids, sort=True),
            "comparison_budget_sha256": provenance["comparison_budget"]["sha256"],
            "prototype_topology_id": provenance["prototype_topology"]["id"],
            "prototype_topology_descriptor_sha256": provenance["prototype_topology"]["descriptor_sha256"],
            "prototype_topology_audit_sha256": provenance["prototype_topology"]["audit_sha256"],
        },
    }


def _records() -> dict:
    samples = len(PATTERNS) * 15
    sample_indices = torch.arange(samples).div(len(PATTERNS), rounding_mode="floor")
    labels = sample_indices.remainder(64)
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
    weather = [weather_names[int(sample_indices[index]) % 3] for index in range(samples)]
    domains = [f"{weather[index]}/scene{int(sample_indices[index]) % 15}" for index in range(samples)]
    patterns = [PATTERNS[index % len(PATTERNS)] for index in range(samples)]
    mask_groups = [
        "full" if name == "full" else "single" if name.endswith("_only") else "drop1" if name.count("_") == 1 else "drop2"
        for name in patterns
    ]
    source_ids = [f"sample-{int(index)}" for index in sample_indices]
    sample_ids = [
        f"mmw:{weather[index]}:scene{int(sample_indices[index]) % 15}:validation:{source_ids[index]}"
        for index in range(samples)
    ]
    protocol_sample_ids = [f"{domains[index]}:{source_ids[index]}" for index in range(samples)]
    group_ids = [f"trajectory-{int(index) // 2}" for index in sample_indices]
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
        "sample_id": sample_ids,
        "protocol_sample_id": protocol_sample_ids,
        "group_id": group_ids,
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
    binding = {
        "schema_version": 1,
        "checkpoint_sha256": "b" * 64,
        "prototype_topology": {"id": "cyclic_index_v1", "descriptor_sha256": "", "audit_sha256": ""},
        "data_protocol": {"protocol_id": TRAJECTORY_PROTOCOL_MODE},
        "experiment_seed": 1,
        "control_checkpoint_sha256": {},
    }
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
            "evidence_binding": binding,
        },
        path,
    )

    records = _load_reusable_evidence(
        path,
        model=Model(),
        expected_patterns={"full"},
        samples_per_pattern=2,
        expected_controls=set(),
        expected_binding=binding,
    )

    assert records["unimodal_confidence"].shape == (2, 5)
    assert records["unimodal_correct"].dtype == torch.bool

    with pytest.raises(ValueError, match="checkpoint lineage"):
        _load_reusable_evidence(
            path,
            model=Model(),
            expected_patterns={"full"},
            samples_per_pattern=2,
            expected_controls=set(),
            expected_binding={**binding, "checkpoint_sha256": "c" * 64},
        )


def test_evaluation_checkpoint_rejects_posthoc_topology_and_protocol_relabel(tmp_path) -> None:
    class Model(torch.nn.Module):
        training_stage = "stage3_fusion"

        def __init__(self, topology: dict[str, object]) -> None:
            super().__init__()
            self._topology = topology

        def prototype_topology_metadata(self) -> dict[str, object]:
            return dict(self._topology)

    records = _records()
    protocol = _protocol(records, trajectory=True)
    formal = _topology(formal=True)
    config = {"experiment": {"seed": 1}, "data_protocol": protocol}

    def checkpoint(directory, *, topology, checkpoint_protocol):
        path, _ = publish_checkpoint(
            {
                "checkpoint_schema_version": 1,
                "checkpoint_role": "validation_best",
                "epoch": 1,
                "state_dict": {},
                "model_metadata": {
                    "training_stage": "stage3_fusion",
                    "prototype_topology_id": topology["id"],
                    "prototype_topology": topology,
                },
                "data_protocol": checkpoint_protocol,
                "experiment_seed": 1,
            },
            directory,
            "stage3_best.pth",
        )
        return path

    cyclic_path = checkpoint(tmp_path / "cyclic", topology=_topology(), checkpoint_protocol=protocol)
    with pytest.raises(ValueError, match="prototype topology"):
        _load_evaluation_checkpoint(
            Model(formal),
            cyclic_path,
            expected_stage="stage3_fusion",
            device=torch.device("cpu"),
            config=config,
        )

    wrong_protocol = {**protocol, "protocol_fingerprint": "e" * 64}
    wrong_protocol_path = checkpoint(tmp_path / "protocol", topology=formal, checkpoint_protocol=wrong_protocol)
    with pytest.raises(ValueError, match="data protocol"):
        _load_evaluation_checkpoint(
            Model(formal),
            wrong_protocol_path,
            expected_stage="stage3_fusion",
            device=torch.device("cpu"),
            config=config,
        )

    matching_path = checkpoint(tmp_path / "matching", topology=formal, checkpoint_protocol=protocol)
    assert len(
        _load_evaluation_checkpoint(
            Model(formal),
            matching_path,
            expected_stage="stage3_fusion",
            device=torch.device("cpu"),
            config=config,
        )
    ) == 64


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
        data_protocol=_protocol(records),
        prototype_topology=_topology(),
        experiment_seed=1,
        validation_identity_binding=_identity_binding(records),
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
        data_protocol=_protocol(records),
        prototype_topology=_topology(),
        experiment_seed=1,
        validation_identity_binding={"sample_count": 14_625, "sample_id_sha256": "f" * 64},
    )

    assert report["stage2_gate_passed"] is False
    assert "bounded_evaluation_not_gate_eligible" in report["failure_reasons"]
    assert report["validation_identity"]["protocol_binding_complete"] is False


def test_matrix_reuses_same_unimodal_probabilities_for_replacements() -> None:
    records = _records()
    confidence, _ = fit_train_confidence_p90(records)
    provenance = _provenance(records)

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


def test_matrix_rejects_mask_sample_reordering() -> None:
    records = _records()
    records["sample_id"][1] = records["sample_id"][len(PATTERNS)]
    confidence, _ = fit_train_confidence_p90(records)

    with pytest.raises(ValueError, match="not paired to the Full-mask sample order"):
        summarize_pcpf_matrix(records, train_confidence_p90=confidence, provenance=_provenance(records))


def test_sparse_csi_matrix_uses_exact_r_d_contracts() -> None:
    records = _five_modality_records()
    train_records = _five_modality_records(pattern_limit="full")
    confidence, _ = fit_train_confidence_p90(train_records)
    provenance = _provenance(records, trajectory=True, comparison_budget=True)

    report = summarize_pcpf_matrix(
        records,
        train_confidence_p90=confidence,
        provenance=provenance,
        diagnostics_config={
            "bootstrap": {"seed": 1, "resamples": 8, "confidence": 0.95},
            "trajectory_r0_reference_summary": _trajectory_r0_summary(records, provenance),
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

    mismatched_r0 = _trajectory_r0_summary(records, provenance)
    mismatched_r0["comparison_contract"]["experiment_seed"] = 2
    with pytest.raises(ValueError, match="experiment_seed"):
        _r0_r7_summary(report, trajectory_r0_reference=mismatched_r0)

    bounded_report = dict(report)
    bounded_report["bounded_evaluation"] = True
    assert _r0_r7_summary(bounded_report)["status"] == "bounded_evaluation_has_no_r0_comparison"


def test_trajectory_r0_loader_requires_hashed_same_protocol_four_modality_report(tmp_path) -> None:
    budget = _comparison_budget_descriptor()
    report = {
        "report_type": "pcpf_15_mask_diagnostics",
        "source_split": "validation",
        "train_confidence_source_split": "train",
        "bounded_evaluation": False,
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "training_stage": "stage3_fusion",
        "modalities": MODALITIES,
        "experiment_seed": 1,
        "expert_fingerprint": "a" * 64,
        "provenance": {
            "checkpoint": {"path": "/tmp/r0.pth", "sha256": "b" * 64, "role": "validation_best"},
            "data_protocol": {
                "mode": TRAJECTORY_PROTOCOL_MODE,
                    "protocol_id": TRAJECTORY_PROTOCOL_MODE,
                    "protocol_fingerprint": "d" * 64,
                    "audit_id": "mmw_id_stratified_block_audit_v1",
                    "audit_sha256": "a" * 64,
                "train_role": "train",
                "validation_role": "validation",
                "validation_sample_count": 14_625,
                "validation_sample_id_hash": "e" * 64,
            },
            "normalization": {"source_split": "train"},
            "comparison_budget": budget,
            "prototype_topology": _topology(formal=True),
        },
        "validation_identity": {
            "source_split": "validation",
            "experiment_seed": 1,
            "sample_count": 14_625,
            "ordered_sample_id_sha256": "f" * 64,
            "protocol_sample_id_sha256": "e" * 64,
            "bound_sample_id_sha256": "e" * 64,
            "outer_test_accessed": False,
        },
        "overall": {"replacement_metrics": {"pcpf_analytic": {"count": 14_625, "top1": 0.5}}},
        "patterns": {"full": {"replacement_metrics": {"pcpf_analytic": {"count": 14_625, "top1": 0.6}}}},
        "pattern_aggregates": {"all14": {"pcpf_analytic": {"top1": {"macro": 0.4}}}},
    }
    path = tmp_path / "trajectory_r0.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    summary = _load_trajectory_r0_reference({"path": str(path), "sha256": digest})

    assert summary["status"] == "verified_same_protocol_trajectory_reference"
    assert summary["report"]["sha256"] == digest
    assert summary["overall15"]["count"] == 14_625
    assert summary["comparison_contract"]["comparison_budget_sha256"] == budget["sha256"]

    report["modalities"] = [*MODALITIES, "csi"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="four-modality"):
        _load_trajectory_r0_reference(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )


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
    protocol_sample_ids = []
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
            protocol_sample_ids.append(f"{domains[-1]}:sample-{sample}")
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
        "protocol_sample_id": protocol_sample_ids,
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


def test_registered_r0_and_sparse_csi_templates_share_the_same_budget() -> None:
    r0 = load_config("tools/configs/pcpf/trajectory_r0/stage3.yaml")
    sparse = load_config("tools/configs/pcpf/sparse_csi/stage3.yaml")

    r0_budget = _comparison_budget(r0)
    sparse_budget = _comparison_budget(sparse)

    assert r0_budget == sparse_budget
    assert r0_budget is not None
    assert len(r0_budget["sha256"]) == 64
    assert _comparison_budget(load_config("tools/configs/pcpf/stage3.yaml")) is None

    r0["training"]["epochs"] = 9
    with pytest.raises(ValueError, match="does not match"):
        _comparison_budget(r0)


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
            "protocol_id": TRAJECTORY_PROTOCOL_MODE,
            "protocol_fingerprint": "a" * 64,
            "train_role": "train",
            "validation_role": "validation",
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
