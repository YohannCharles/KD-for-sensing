import hashlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.eval.pcpf import (
    _protocol_sample_id,
    build_stage2_gate_report,
    fit_train_confidence_p90,
    resolve_pcpf_missing_patterns,
    summarize_pcpf_matrix,
)
from kd_sensing.models.pcpf_temporal_risk import analytic_fusion_weights
from kd_sensing.utils.checkpoint import publish_checkpoint
from tools.eval_pcpf import (
    _evaluation_normalization_metadata,
    _load_evaluation_checkpoint,
    _load_reusable_evidence,
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


def test_protocol_sample_id_preserves_audited_mmw_identity() -> None:
    sample_id = "foggy:Town03:Town03_5wayroad_seed28:cav_1:000849"

    assert _protocol_sample_id({"source_sample_id": sample_id}) == sample_id
    assert _protocol_sample_id({"sample_id": sample_id}) == sample_id
    with pytest.raises(ValueError, match="audited protocol sample identity"):
        _protocol_sample_id({})


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


def _topology(*, formal: bool = False) -> dict[str, object]:
    return {
        "id": "ula_dft_phase_cycle_v1" if formal else "cyclic_index_v1",
        "descriptor_sha256": "1" * 64 if formal else "",
        "audit_path": "/tmp/topology.json" if formal else "",
        "audit_sha256": "2" * 64 if formal else "",
    }


def _provenance(records: dict, *, trajectory: bool = False) -> dict:
    return {
        "checkpoint": {"path": "/tmp/stage3_best.pth", "sha256": "c" * 64, "role": "validation_best"},
        "data_protocol": _protocol(records, trajectory=trajectory),
        "normalization": {
            "source_split": "train",
            "risk_component_std": [0.01, 0.2, 0.3, 0.4],
        },
        "experiment_seed": 1,
        "prototype_topology": _topology(formal=trajectory),
        "validation_identity_binding": _identity_binding(records),
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
    }
    torch.save(
        {
            "training_stage": "stage1_expert",
            "modalities": list(Model.modalities),
            "expert_fingerprint": "a" * 64,
            "bounded_evaluation": False,
            "pattern": ["full", "full"],
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
    protocol = {
        **_protocol(records),
        "protocol_version": 1,
        "block_size": 128,
        "split_manifest_hash": "1" * 64,
        "data_source_hash": "2" * 64,
        "window_config_hash": "3" * 64,
        "weather_binding": True,
        "train_block_count": 90,
        "validation_block_count": 19,
        "test_block_count": 19,
    }

    report = build_stage2_gate_report(
        records,
        _gate(),
        train_confidence_p90=confidence,
        train_confidence_count=count,
        stage2_checkpoint_sha256="b" * 64,
        bounded_evaluation=False,
        data_protocol=protocol,
        prototype_topology=_topology(),
        experiment_seed=1,
        validation_identity_binding=_identity_binding(records),
    )

    assert report["stage2_gate_passed"] is True
    assert report["failure_reasons"] == []
    assert report["overall"]["spearman"] > 0.99
    assert len(report["domains"]) == 15
    assert set(report["mask_groups"]) == {"full", "drop1", "drop2", "single"}
    assert all(report["data_protocol"][key] == protocol[key] for key in protocol)


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
    assert report["evaluation_scope"] == "checkpoint_bound_matrix"
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
    provenance = _provenance(records, trajectory=True)

    report = summarize_pcpf_matrix(
        records,
        train_confidence_p90=confidence,
        provenance=provenance,
        diagnostics_config={"bootstrap": {"seed": 1, "resamples": 8, "confidence": 0.95}},
    )

    assert len(report["patterns"]) == 31
    assert report["pattern_aggregates"]["all30"]["pcpf_analytic"]["top1"]["pattern_count"] == 30
    assert report["pattern_aggregates"]["csi_present_with_sensing"]["pcpf_analytic"]["top1"]["pattern_count"] == 15
    assert report["pattern_aggregates"]["csi_absent_legacy15"]["pcpf_analytic"]["top1"]["pattern_count"] == 15
    assert len(report["pattern_domain"]) == 31 * 3
    assert set(report["expert_input_diagnostics"]["prototype_distance"]) == set(records["modalities"])
    assert report["evaluation_scope"] == "checkpoint_bound_matrix"
    assert set(report["overall"]["replacement_metrics"]) == {"uniform", "static_prior", "pcpf_analytic"}
    methods = report["mechanism_diagnostics"]["dynamicity_test"]["methods"]
    assert set(methods) == {
        "D0_original_sample_risk",
        "D1_domain_mask_shuffled_risk",
        "D2_domain_mask_mean_risk",
        "D3_static_prior",
    }



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
        },
        "replacement_weights": {
            "uniform": uniform,
            "static_prior": static,
            "pcpf_analytic": dynamic,
        },
        "modality_temperatures": torch.ones(5),
        "static_capability": capability,
        "fusion_tau": 1.0,
        "risk_coefficients": torch.ones(4),
        "risk_component_mean": torch.zeros(4),
        "risk_component_std": torch.ones(4),
        "expert_fingerprint": "e" * 64,
        "training_stage": "stage3_fusion",
        "bounded_evaluation": False,
    }


def test_matrix_rejects_missing_provenance() -> None:
    records = _records()
    confidence, _ = fit_train_confidence_p90(records)

    with pytest.raises(ValueError, match="data_protocol, normalization"):
        summarize_pcpf_matrix(records, train_confidence_p90=confidence, provenance={"checkpoint": {}})
