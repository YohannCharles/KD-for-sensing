import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kd_sensing.config import load_config
from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.models.pcpf_temporal_risk import PCPFTemporalRiskFusion
from kd_sensing.models.u_mask_beam_jepa import UMaskBeamJEPA
from kd_sensing.registries import ENCODERS
from kd_sensing.utils.checkpoint import checkpoint_file_digest


@ENCODERS.register("pcpf_workflow_test_sequence", force=True)
class _WorkflowSequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 64, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.scale = nn.Parameter(torch.ones(self.output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            scalar = value.float().mean(dim=-1, keepdim=True)
        else:
            scalar = value.float().flatten(start_dim=2).mean(dim=-1, keepdim=True)
        return scalar * self.scale.view(1, 1, -1)


class _WorkflowDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(5, 3, 8, 8, generator=generator),
            "radar_ra": torch.randn(5, 1, 128, 64, generator=generator),
            "radar_da": torch.randn(5, 1, 128, 64, generator=generator),
            "gps": torch.randn(5, 3, generator=generator),
            "lidar": torch.randn(5, 3, 8, 8, generator=generator),
            "target_beam": torch.tensor([index]),
        }


def _encoders() -> dict[str, dict[str, object]]:
    return {name: {"type": "pcpf_workflow_test_sequence", "output_dim": 64} for name in ("image", "radar", "gps", "lidar")}


def _model(stage: str) -> PCPFTemporalRiskFusion:
    return PCPFTemporalRiskFusion(
        encoders=_encoders(),
        training_stage=stage,
        fusion_mode="uniform" if stage == "stage2_risk" else "pcpf_analytic",
        temporal_transformer={"dropout": 0.0},
    )


def test_train_only_stage_preparation_fits_normalization_and_confidence(tmp_path) -> None:
    model = _model("stage2_risk")
    loader = DataLoader(_WorkflowDataset(), batch_size=2, shuffle=True)
    cfg = {
        "experiment": {"task": "fusion"},
        "model": {"primary": {"training_stage": "stage2_risk"}},
        "data_protocol": {"mode": TRAJECTORY_PROTOCOL_MODE, "train_role": "train"},
        "data": {"dataloader": {"train_batch_size": 2}},
        "training": {"resume": False},
        "loss": {
            "pcpf_temporal_risk": {
                "enabled": True,
                "prototype_topology": "cyclic_index_v1",
                "stage_preparation": {"enabled": True, "max_batches": 1, "smoke_only": True},
            }
        },
    }

    report = model.prepare_training_stage(
        cfg=cfg,
        train_loader=loader,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        non_blocking=False,
    )

    assert report["source_split"] == "train"
    assert report["outer_test_accessed"] is False
    assert bool(model.risk_stats_fitted.item())
    assert model.risk_component_count.gt(0).all()
    assert model.risk_component_std[0].item() == pytest.approx(0.01)
    assert model.train_confidence_count.gt(0).all()
    assert model.train_confidence_p90.gt(0).all()

    batch = next(iter(DataLoader(_WorkflowDataset(), batch_size=4, shuffle=False)))
    labels = prepare_task_labels(batch, num_pred=1, device=torch.device("cpu"))
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=5e-4)
    step = run_model_step(
        model,
        "fusion",
        batch,
        seq_length=5,
        num_pred=1,
        device=torch.device("cpu"),
    )
    loss = model.compute_validation_loss(step.model_output, labels, cfg)
    optimizer.zero_grad()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
    optimizer.step()

    model.eval()
    updated = run_model_step(
        model,
        "fusion",
        batch,
        seq_length=5,
        num_pred=1,
        device=torch.device("cpu"),
    ).model_output.diagnostics
    available_risk = updated["raw_risk"][updated["available_modalities"]]
    assert torch.isfinite(available_risk).all()
    assert available_risk.gt(0).all()
    assert available_risk.std().item() > 1e-4


def test_formal_topology_revalidates_the_bound_audit_file(tmp_path) -> None:
    descriptor = {
        "topology_id": "ula_dft_phase_cycle_v1",
        "codebook_type": "ula_dft",
        "num_beams": 64,
    }
    descriptor_sha256 = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit_path = tmp_path / "topology.json"
    audit_path.write_text(
        json.dumps({"descriptor": descriptor, "descriptor_sha256": descriptor_sha256}),
        encoding="utf-8",
    )
    audit_sha256, _ = checkpoint_file_digest(audit_path)

    model = PCPFTemporalRiskFusion(
        encoders=_encoders(),
        training_stage="stage2_risk",
        fusion_mode="uniform",
        temporal_transformer={"dropout": 0.0},
        prototype_topology_id="ula_dft_phase_cycle_v1",
        prototype_topology_descriptor_sha256=descriptor_sha256,
        prototype_topology_audit_path=str(audit_path),
        prototype_topology_audit_sha256=audit_sha256,
    )
    assert model.prototype_topology_metadata()["formal_r0_r7_eligible"] is True

    audit_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="audit SHA256"):
        PCPFTemporalRiskFusion(
            encoders=_encoders(),
            training_stage="stage2_risk",
            fusion_mode="uniform",
            temporal_transformer={"dropout": 0.0},
            prototype_topology_id="ula_dft_phase_cycle_v1",
            prototype_topology_descriptor_sha256=descriptor_sha256,
            prototype_topology_audit_path=str(audit_path),
            prototype_topology_audit_sha256=audit_sha256,
        )


def test_stage3_gate_binds_sha_unbounded_report_and_expert_fingerprint(tmp_path) -> None:
    model = _model("stage3_fusion")
    protocol = {
        "protocol_id": TRAJECTORY_PROTOCOL_MODE,
        "protocol_fingerprint": "d" * 64,
        "train_role": "train",
        "validation_role": "validation",
        "validation_sample_count": 14_625,
        "validation_sample_id_hash": "e" * 64,
    }
    report = {
        "stage2_gate_passed": True,
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "source_training_stage": "stage2_risk",
        "bounded_evaluation": False,
        "expert_fingerprint": model._expert_fingerprint(),
        "prototype_topology": model.prototype_topology_metadata(),
        "data_protocol": dict(protocol),
        "source_split": "validation",
        "train_confidence_source_split": "train",
        "experiment_seed": 1,
        "validation_identity": {
            "sample_count": 14_625,
            "protocol_sample_id_sha256": "e" * 64,
            "bound_sample_id_sha256": "e" * 64,
        },
        "stage2_checkpoint_sha256": "f" * 64,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    digest, _ = checkpoint_file_digest(path)
    cfg = {
        "experiment": {"seed": 1},
        "data_protocol": protocol,
        "training": {
            "initialization_checkpoint": {"sha256": "f" * 64},
            "pcpf_stage2_gate": {
                "report_path": str(path),
                "sha256": digest,
                "stage2_gate_passed": True,
            },
        },
    }

    model._validate_stage2_gate_binding(cfg)

    report["expert_fingerprint"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    cfg["training"]["pcpf_stage2_gate"]["sha256"] = checkpoint_file_digest(path)[0]
    with pytest.raises(ValueError, match="expert fingerprint"):
        model._validate_stage2_gate_binding(cfg)

    report["expert_fingerprint"] = model._expert_fingerprint()
    report["data_protocol"]["protocol_fingerprint"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    cfg["training"]["pcpf_stage2_gate"]["sha256"] = checkpoint_file_digest(path)[0]
    with pytest.raises(ValueError, match="data protocol"):
        model._validate_stage2_gate_binding(cfg)


def test_u0_state_dict_remains_free_of_pcpf_owners() -> None:
    model = UMaskBeamJEPA(
        encoders=_encoders(),
        temporal_pooling={"enabled": True, "type": "masked_mean"},
    )
    keys = set(model.state_dict())

    assert not any(key.startswith("probability_head.") for key in keys)
    assert "risk_coefficient_raw" not in keys
    assert "temperature_raw" not in keys
    assert "tau_raw" not in keys
    assert "train_confidence_p90" not in keys


def test_pcpf_config_keeps_image_profile_at_dataset_boundary() -> None:
    cfg = load_config("tools/configs/pcpf/stage1.yaml")

    assert cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert cfg["model"]["primary"]["risk"]["normalization_epsilon"] == pytest.approx(0.01)
    assert cfg["training"]["checkpoint_selection"] == "best_validation_loss"
    assert "image_profile" not in cfg["model"]["primary"]


def test_direct_router_control_allows_only_its_new_checkpoint_prefix() -> None:
    cfg = load_config("tools/configs/pcpf/ablations/a2_direct_router_control.yaml")

    initialization = cfg["training"]["initialization_checkpoint"]
    assert initialization["allowed_missing_prefixes"] == ["direct_router"]


def test_all_pcpf_templates_pass_strict_config_loading() -> None:
    for path in sorted(Path("tools/configs/pcpf").rglob("*.yaml")):
        cfg = load_config(path)
        assert cfg["model"]["primary"]["type"] == "pcpf_temporal_risk_fusion"
