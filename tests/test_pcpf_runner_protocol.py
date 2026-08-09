import copy
import hashlib
import json
from pathlib import Path

import pytest

import tools.run_pcpf as run_pcpf
from kd_sensing.data.mmw.trajectory_protocol import (
    ASSIGNMENT_ALGORITHM,
    DEFAULT_BLOCK_SIZE,
    TRAJECTORY_MANIFEST_VERSION,
    TRAJECTORY_PROTOCOL_MODE,
)
from kd_sensing.utils.checkpoint import publish_checkpoint
from tools.run_pcpf import (
    _apply_train_seed,
    _bind_gate,
    _bind_topology_audit,
    _checkpoint_experiment_seed,
    _checkpoint_protocol_fingerprint,
    _completed_stage_checkpoint,
    _configure_single_modality_diagnostic,
    _continuation_stage_templates,
    _configured_topology,
    _load_template,
    _next_stage_run_name,
    _run_stage2_gate,
    _validate_continuation_binding,
    _validate_trajectory_audit,
    _validate_reusable_gate_report,
)


TRAJECTORY_FINGERPRINT = "d" * 64


def test_one_batch_smoke_cli_forwards_topology_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_pcpf, "real_one_batch_smoke", lambda *args, **kwargs: captured.update(kwargs) or {})
    topology_audit = tmp_path / "topology.json"

    assert run_pcpf.main(
        ["one-batch-smoke", "--topology-audit", str(topology_audit), "--batch-size", "64"]
    ) == 0
    assert captured["topology_audit"] == topology_audit
    assert captured["batch_size"] == 64


def test_single_modality_diagnostic_is_stage1_only_and_disables_mask_matrix() -> None:
    cfg = {
        "experiment": {},
        "model": {
            "primary": {
                "modalities": ["image", "radar", "gps", "lidar"],
                "use_sparse_csi": True,
            }
        },
        "temporal_missing": {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "schedule_id": "pcpf_five_modality_all_subsets_v1",
            "panel_size": 31,
        },
        "loss": {"pcpf_temporal_risk": {"lambda_unimodal": 5.0}},
        "evaluation": {"missing_patterns": {"enabled": True, "patterns": ["all_nonempty"]}},
    }

    assert _configure_single_modality_diagnostic(cfg, stage="stage1", modality="csi") == "csi"
    assert cfg["temporal_missing"] == {
        "enabled": True,
        "mode": "fixed_single_modality",
        "fixed_modality": "csi",
    }
    assert cfg["evaluation"]["missing_patterns"]["enabled"] is False
    assert cfg["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 1.0
    assert cfg["experiment"]["single_modality_diagnostic"]["stage1_only"] is True

    with pytest.raises(ValueError, match="fresh-start Stage 1"):
        _configure_single_modality_diagnostic(cfg, stage="stage2", modality="image")
    with pytest.raises(ValueError, match="must be one of"):
        _configure_single_modality_diagnostic(cfg, stage="stage1", modality="audio")


def _protocol() -> dict[str, object]:
    return {
        "mode": TRAJECTORY_PROTOCOL_MODE,
        "protocol_id": TRAJECTORY_PROTOCOL_MODE,
        "protocol_version": 1,
        "manifest_version": TRAJECTORY_MANIFEST_VERSION,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "protocol_fingerprint": TRAJECTORY_FINGERPRINT,
        "split_manifest_hash": "b" * 64,
        "data_source_hash": "c" * 64,
        "window_config_hash": "e" * 64,
        "block_size": DEFAULT_BLOCK_SIZE,
        "weather_binding": True,
        "audit_id": "mmw_id_stratified_block_audit_v1",
        "audit_sha256": "a" * 64,
        "split_seed": 0,
        "train_role": "train",
        "validation_role": "validation",
        "train_window_count": 100,
        "validation_window_count": 20,
        "test_window_count": 20,
        "train_block_count": 70,
        "validation_block_count": 15,
        "test_block_count": 15,
        "trajectory_count": 16,
    }


def _audit() -> dict[str, object]:
    return {
        "audit_id": "trajectory-audit",
        "status": "passed",
        "test_evaluated": False,
        "protocol": TRAJECTORY_PROTOCOL_MODE,
        "protocol_version": 1,
        "manifest_version": TRAJECTORY_MANIFEST_VERSION,
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "split_seed": 0,
        "block_size": DEFAULT_BLOCK_SIZE,
        "split_manifest_hash": "b" * 64,
        "data_source_hash": "c" * 64,
        "window_config_hash": "e" * 64,
        "weather_binding": True,
        "protocol_fingerprint": TRAJECTORY_FINGERPRINT,
        "train_sample_count": 100,
        "validation_sample_count": 20,
        "test_sample_count": 20,
        "block_counts": {"train": 70, "validation": 15, "test": 15},
        "trajectory_counts": {"train": 16, "validation": 16, "test": 16},
        "checks": {"block_overlap": True, "weather_binding": True, "window_crossing": True},
    }


def test_trajectory_audit_requires_exact_counts_and_passed_structural_checks(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(_audit()), encoding="utf-8")

    validated = _validate_trajectory_audit(path, _protocol())

    assert validated["train_sample_count"] == 100
    assert validated["validation_sample_count"] == 20
    assert validated["test_evaluated"] is False

    failed = _audit()
    failed["checks"]["weather_binding"] = False
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="pass every"):
        _validate_trajectory_audit(path, _protocol())


def test_checkpoint_protocol_fingerprint_supports_formal_and_smoke_payloads() -> None:
    assert (
        _checkpoint_protocol_fingerprint(
            {"resume_contract": {"config": {"data_protocol": {"protocol_fingerprint": TRAJECTORY_FINGERPRINT}}}}
        )
        == TRAJECTORY_FINGERPRINT
    )
    assert _checkpoint_protocol_fingerprint({"data_protocol": {"protocol_fingerprint": TRAJECTORY_FINGERPRINT}}) == TRAJECTORY_FINGERPRINT
    assert _checkpoint_protocol_fingerprint({}) is None
    assert _checkpoint_experiment_seed({"experiment_seed": 7}) == 7
    assert _checkpoint_experiment_seed({"resume_contract": {"config": {"experiment": {"seed": 9}}}}) == 9
    with pytest.raises(ValueError, match="experiment seed provenance"):
        _checkpoint_experiment_seed({})


def test_sparse_trajectory_stage1_template_is_fresh_start() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = _load_template(root / "tools/configs/pcpf/sparse_csi/stage1.yaml")

    assert cfg["training"]["initialization_checkpoint"] is False
    assert "historical_reference" not in cfg["evaluation"]["pcpf_diagnostics"]
    assert cfg["data"]["dataloader"]["train_batch_size"] == 64
    assert cfg["data"]["dataloader"]["validation_batch_size"] == 64
    assert cfg["data"]["dataloader"]["num_workers"] == 8


def test_topology_loss_off_template_only_disables_topology_supervision() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = _load_template(root / "tools/configs/pcpf/sparse_csi/stage1.yaml")
    ablation = _load_template(root / "tools/configs/pcpf/sparse_csi/stage1_no_topology_loss.yaml")
    baseline_loss = baseline["loss"]["pcpf_temporal_risk"]
    ablation_loss = ablation["loss"]["pcpf_temporal_risk"]

    assert ablation_loss["unimodal_soft_weight"] == 0.0
    assert ablation_loss["use_beam_prototype_alignment"] is False
    assert ablation_loss["lambda_proto"] == 0.0
    assert ablation_loss["lambda_modality_proto"] == 0.0
    for key in ("lambda_fused_hard", "lambda_unimodal", "unimodal_hard_weight", "fused_soft_weight"):
        assert ablation_loss[key] == baseline_loss[key]
    assert ablation_loss["prototype_topology"] == baseline_loss["prototype_topology"]
    assert ablation["model"] == baseline["model"]
    assert ablation["temporal_missing"] == baseline["temporal_missing"]
    assert ablation["training"] == baseline["training"]


def test_weak_expert_screen_templates_change_only_the_registered_factor() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = _load_template(root / "tools/configs/pcpf/sparse_csi/stage1.yaml")
    j0 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/j0_joint_bf16_control.yaml")
    j1 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/j1_joint_rebalanced.yaml")
    j1_no_topology = _load_template(
        root / "tools/configs/pcpf/sparse_csi/weak_experts/j1_joint_rebalanced_no_topology.yaml"
    )
    j2 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/j2_joint_csi_spatial4x2_layer1.yaml")
    j2_stage2 = _load_template(
        root / "tools/configs/pcpf/sparse_csi/weak_experts/j2_joint_csi_spatial4x2_layer1_stage2.yaml"
    )
    j2_stage3 = _load_template(
        root / "tools/configs/pcpf/sparse_csi/weak_experts/j2_joint_csi_spatial4x2_layer1_stage3.yaml"
    )
    c1 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/c1_csi_token_layer1.yaml")
    c2 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/c2_csi_spatial4x2_layer1.yaml")
    r1 = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/r1_radar_dual_branch.yaml")

    assert baseline["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 1.0
    assert j0["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 1.0
    assert j1["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 5.0
    assert j1_no_topology["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 5.0
    assert c1["model"]["primary"]["sparse_csi_encoder"]["num_layers"] == 1
    assert c2["model"] == c1["model"]
    assert c2["loss"] == c1["loss"]
    assert c2["training"] == c1["training"]
    assert c2["temporal_missing"] == c1["temporal_missing"]
    c1_sparse = c1["data"]["dataset"]["sparse_csi"]
    c2_sparse = c2["data"]["dataset"]["sparse_csi"]
    assert c2_sparse["selection_sha256"] == "2d035d64f6b9ac408532040b3ff09151a8831361d81c83b1b77e218e4344a4f4"
    assert c2_sparse["cache_manifest_path"].endswith("trajectory_cache_manifest_4x2.json")
    assert set(c2_sparse) - set(c1_sparse) == {"cache_manifest_path"}
    assert all(c2_sparse[key] == value for key, value in c1_sparse.items() if key != "selection_sha256")
    assert baseline["model"]["primary"]["sparse_csi_encoder"]["num_layers"] == 0
    assert r1["model"]["primary"]["encoders"]["radar"]["type"] == "radar_dual_branch_cnn"
    assert baseline["model"]["primary"]["encoders"]["radar"]["type"] == "radar_cnn"
    assert j0["training"]["amp"] == j1["training"]["amp"] == {
        "enabled": True,
        "dtype": "bfloat16",
        "grad_scaler": False,
    }
    assert j1_no_topology["training"]["amp"] == j1["training"]["amp"]
    assert j0["model"] == j1["model"]
    assert j0["temporal_missing"] == j1["temporal_missing"]
    assert j2["training"] == j0["training"]
    assert j2["loss"] == j0["loss"]
    assert j2["temporal_missing"] == j0["temporal_missing"]
    assert j2["model"]["primary"]["sparse_csi_encoder"]["num_layers"] == 1
    assert j2["data"]["dataset"]["sparse_csi"] == c2["data"]["dataset"]["sparse_csi"]
    assert j2["data"]["dataloader"]["generator_seeds"] == {
        "train": 3702095051185301119,
        "validation": 5941928843505026558,
    }
    for continuation in (j2_stage2, j2_stage3):
        assert continuation["data"]["dataset"]["sparse_csi"] == j2["data"]["dataset"]["sparse_csi"]
        assert continuation["data"]["dataloader"]["generator_seeds"] == j2["data"]["dataloader"]["generator_seeds"]
        assert continuation["model"]["primary"]["sparse_csi_encoder"] == j2["model"]["primary"]["sparse_csi_encoder"]

    _configure_single_modality_diagnostic(j1, stage="stage1", modality="csi")
    assert j1["loss"]["pcpf_temporal_risk"]["lambda_unimodal"] == 1.0


def test_stage2_gate_uses_checkpoint_run_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_config = tmp_path / "stage2_pre_training.yaml"
    checkpoint = tmp_path / "stage2_run" / "checkpoints" / "stage2_best.pth"
    output = tmp_path / "gate.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(FileNotFoundError, match="run-local resolved config"):
        _run_stage2_gate(requested_config, checkpoint, output, device_name=None)

    run_config = checkpoint.parent.parent / "resolved_config.yaml"
    run_config.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("tools.run_pcpf._run_child", lambda command: commands.append(command) or 0)

    assert _run_stage2_gate(requested_config, checkpoint, output, device_name="cuda") == 0
    config_index = commands[0].index("--config") + 1
    assert commands[0][config_index] == str(run_config.resolve())
    assert commands[0][-2:] == ["--device", "cuda"]


def test_stage3_gate_binding_rejects_protocol_split_seed_identity_or_checkpoint_drift(tmp_path: Path) -> None:
    protocol = {
        "protocol_id": TRAJECTORY_PROTOCOL_MODE,
        "protocol_fingerprint": TRAJECTORY_FINGERPRINT,
        "train_role": "train",
        "validation_role": "validation",
        "validation_sample_count": 14_625,
        "validation_sample_id_hash": "e" * 64,
    }
    cfg = {
        "experiment": {"seed": 1},
        "model": {"primary": {"prototype_topology_id": "cyclic_index_v1"}},
        "loss": {"pcpf_temporal_risk": {"prototype_topology": {"id": "cyclic_index_v1"}}},
        "data_protocol": protocol,
        "training": {"initialization_checkpoint": {"sha256": "b" * 64}},
    }
    report = {
        "stage2_gate_passed": True,
        "bounded_evaluation": False,
        "source_training_stage": "stage2_risk",
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "prototype_topology": {
            "id": "cyclic_index_v1",
            "descriptor_sha256": "",
            "audit_sha256": "",
        },
        "data_protocol": dict(protocol),
        "source_split": "validation",
        "train_confidence_source_split": "train",
        "experiment_seed": 1,
        "validation_identity": {
            "sample_count": 14_625,
            "protocol_sample_id_sha256": "e" * 64,
            "bound_sample_id_sha256": "e" * 64,
        },
        "stage2_checkpoint_sha256": "b" * 64,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    _validate_reusable_gate_report(report, cfg, checkpoint_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="checkpoint/config lineage"):
        _validate_reusable_gate_report(report, cfg, checkpoint_sha256="c" * 64)

    _bind_gate(cfg, path)

    assert cfg["training"]["pcpf_stage2_gate"]["stage2_gate_passed"] is True
    assert len(cfg["training"]["pcpf_stage2_gate"]["sha256"]) == 64

    report["source_split"] = "test"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="split or seed"):
        _bind_gate(cfg, path)


def test_topology_binding_requires_complete_64_beam_trajectory_audit(tmp_path: Path) -> None:
    descriptor = {
        "audit_version": "mmw_ula_dft_codebook_topology_v1",
        "topology_id": "ula_dft_phase_cycle_v1",
        "codebook_type": "ula_dft",
        "num_beams": 64,
        "num_antennas": 64,
        "codebook_sha256": "a" * 64,
        "endpoint_labels": [0, 63],
        "endpoint_phase_gap_bins": 1 / 64,
        "endpoint_u_error_p95": 0.01,
        "power_replay_top1_agreement": True,
        "power_replay_max_abs_error": 1e-12,
        "claim_boundary": "local_ula_dft_phase_codebook_not_world_azimuth_ring",
    }
    domains = [{"id": f"weather/scene-{index}"} for index in range(15)]
    payload = {
        "schema_version": 1,
        "audit_version": "mmw_ula_dft_codebook_topology_v1",
        "descriptor": descriptor,
        "descriptor_sha256": hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "domain_count": 15,
        "domains": [{"id": item["id"], "metadata_status": "verified"} for item in domains],
        "metadata_consistent": True,
        "errors": [],
        "label_table": [{"label": index, "phase_coordinate": index / 64} for index in range(64)],
        "edge_count": 64,
        "power_replay_count": 15,
        "frame_audit_count": 15,
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = {
        "model": {"primary": {"prototype_topology_id": "cyclic_index_v1"}},
        "loss": {"pcpf_temporal_risk": {"prototype_topology": {"id": "cyclic_index_v1"}}},
    }

    binding = _bind_topology_audit(cfg, path, _protocol(), domains)

    assert binding["id"] == "ula_dft_phase_cycle_v1"
    assert binding["descriptor_sha256"] == payload["descriptor_sha256"]
    assert len(binding["audit_sha256"]) == 64
    assert cfg["model"]["primary"]["prototype_topology_audit_sha256"] == binding["audit_sha256"]
    assert cfg["loss"]["pcpf_temporal_risk"]["prototype_topology"]["audit_path"] == str(path)

    payload["label_table"][0]["phase_coordinate"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label order"):
        _bind_topology_audit(cfg, path, _protocol(), domains)

    payload["label_table"][0]["phase_coordinate"] = 0.0
    last_domain = payload["domains"][-1]
    payload["domains"][-1] = dict(payload["domains"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate trajectory domains"):
        _bind_topology_audit(cfg, path, _protocol(), domains)

    payload["domains"][-1] = last_domain
    payload["metadata_consistent"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        _bind_topology_audit(cfg, path, _protocol(), domains)


def test_configured_formal_topology_cannot_be_declared_by_hash_shape_alone() -> None:
    cfg = {
        "model": {
            "primary": {
                "prototype_topology_id": "ula_dft_phase_cycle_v1",
                "prototype_topology_descriptor_sha256": "1" * 64,
                "prototype_topology_audit_path": "/definitely/missing.json",
                "prototype_topology_audit_sha256": "2" * 64,
            }
        },
        "loss": {
            "pcpf_temporal_risk": {
                "prototype_topology": {
                    "id": "ula_dft_phase_cycle_v1",
                    "descriptor_sha256": "1" * 64,
                    "audit_path": "/definitely/missing.json",
                    "audit_sha256": "2" * 64,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="data protocol provenance"):
        _configured_topology(cfg)


def test_pipeline_stage_names_and_train_seed_follow_stage1() -> None:
    cfg = {"experiment": {"seed": 1}, "evaluation": {"pcpf_diagnostics": {"bootstrap": {"seed": 1}}}}

    _apply_train_seed(cfg, 7)

    assert cfg["experiment"]["seed"] == 7
    assert cfg["experiment"]["train_seed"] == 7
    assert cfg["evaluation"]["pcpf_diagnostics"]["bootstrap"]["seed"] == 7
    assert _next_stage_run_name("stage1_trajectory_seed7", "stage2") == "stage2_trajectory_seed7"
    assert _next_stage_run_name("custom", "stage3") == "custom_stage3"


def test_continuation_prefers_complete_stage1_specific_template_pair(tmp_path: Path) -> None:
    stage1 = tmp_path / "candidate.yaml"
    generic = (tmp_path / "stage2.yaml", tmp_path / "stage3.yaml")
    specific = (tmp_path / "candidate_stage2.yaml", tmp_path / "candidate_stage3.yaml")
    for path in (stage1, *generic, *specific):
        path.touch()

    assert _continuation_stage_templates(stage1) == specific

    specific[1].unlink()
    with pytest.raises(FileNotFoundError, match="both Stage 2 and Stage 3"):
        _continuation_stage_templates(stage1)


def test_continuation_binding_rejects_j2_sparse_csi_drift() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = _load_template(root / "tools/configs/pcpf/sparse_csi/weak_experts/j2_joint_csi_spatial4x2_layer1.yaml")

    candidate = copy.deepcopy(reference)
    candidate["data"]["dataset"]["sparse_csi"]["selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selection and packed cache"):
        _validate_continuation_binding(reference, candidate)

    candidate = copy.deepcopy(reference)
    candidate["data"]["dataloader"]["generator_seeds"]["train"] += 1
    with pytest.raises(ValueError, match="generator seeds"):
        _validate_continuation_binding(reference, candidate)

    candidate = copy.deepcopy(reference)
    candidate["model"]["primary"]["sparse_csi_encoder"]["num_layers"] = 0
    with pytest.raises(ValueError, match="encoder config"):
        _validate_continuation_binding(reference, candidate)


def test_completed_stage_requires_full_epoch_and_matching_best_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage1_run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    cfg = {
        "experiment": {"seed": 1},
        "model": {"primary": {"training_stage": "stage1_expert"}},
        "training": {"epochs": 2},
        "output": {"dir": str(tmp_path), "run_name": "stage1_run"},
    }

    def publish(
        filename: str,
        *,
        role: str,
        epoch: int,
        stage: str = "stage1_expert",
        recorded_seed: int = 1,
        recorded_epochs: int = 2,
    ) -> None:
        recorded_cfg = {
            **cfg,
            "experiment": {"seed": recorded_seed},
            "training": {key: value for key, value in cfg["training"].items() if key != "epochs"},
        }
        publish_checkpoint(
            {
                "checkpoint_schema_version": 1,
                "checkpoint_role": role,
                "epoch": epoch,
                "model_metadata": {"training_stage": stage},
                "resume_contract": {"config": recorded_cfg, "training_epochs": recorded_epochs},
            },
            checkpoints,
            filename,
        )

    publish("last.pth", role="last", epoch=2)
    publish("stage1_best.pth", role="validation_best", epoch=1)
    assert _completed_stage_checkpoint(cfg) == checkpoints / "stage1_best.pth"

    publish("stage1_best.pth", role="validation_best", epoch=1, recorded_seed=2)
    with pytest.raises(RuntimeError, match="config lineage mismatch"):
        _completed_stage_checkpoint(cfg)
    publish("stage1_best.pth", role="validation_best", epoch=1)

    publish("last.pth", role="last", epoch=2, recorded_epochs=3)
    with pytest.raises(RuntimeError, match="config lineage mismatch"):
        _completed_stage_checkpoint(cfg)

    publish("last.pth", role="last", epoch=1)
    with pytest.raises(RuntimeError, match="full-budget last checkpoint"):
        _completed_stage_checkpoint(cfg)

    publish("last.pth", role="last", epoch=2)
    publish("stage1_best.pth", role="validation_best", epoch=1, stage="stage2_risk")
    with pytest.raises(RuntimeError, match="validation-best checkpoint"):
        _completed_stage_checkpoint(cfg)
