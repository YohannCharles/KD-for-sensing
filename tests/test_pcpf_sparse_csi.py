import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from kd_sensing.channel.probe_codebook import generate_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import load_path_channel, simulate_candidate_pilots
from kd_sensing.data.pcpf_sparse_csi import (
    PCPFSparseCSISidecar,
    PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ,
    PCPF_SPARSE_CSI_SELECTION_SHA256,
)
from kd_sensing.data.temporal_missing import apply_training_temporal_missing
from kd_sensing.engine.training_extensions import EpochDiagnosticsAccumulator
from kd_sensing.models.pcpf_temporal_risk import PCPFTemporalRiskFusion
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder
from kd_sensing.models.temporal_transformer import SharedTemporalTransformer
from kd_sensing.registries import ENCODERS


@ENCODERS.register("pcpf_sparse_csi_test_sequence", force=True)
class _TestSequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 64, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(64, self.output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _encoders() -> dict[str, dict[str, object]]:
    return {
        name: {"type": "pcpf_sparse_csi_test_sequence", "output_dim": 64}
        for name in ("image", "radar", "gps", "lidar")
    }


def _csi_inputs(batch_size: int, *, available: bool = True) -> dict[str, torch.Tensor]:
    real = torch.randn(batch_size, 5, 2, 2)
    imaginary = torch.randn(batch_size, 5, 2, 2)
    return {
        "csi_batch": torch.complex(real, imaginary),
        "csi_pattern_ids": torch.tensor([0, 1]).view(1, 1, 2).expand(batch_size, 5, -1),
        "csi_frequency_positions": torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ),
        "csi_frequency_ids": torch.tensor([0, 15]),
        "csi_pilot_mask": torch.full((batch_size, 5, 2, 2), available, dtype=torch.bool),
        "csi_snr_available": torch.zeros(batch_size, dtype=torch.bool),
    }


def _sensing_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        f"{name}_batch": torch.randn(batch_size, 5, 64)
        for name in ("image", "radar", "gps", "lidar")
    }


def test_sparse_pilot_encoder_marks_missing_snr_without_changing_feature() -> None:
    torch.manual_seed(2)
    encoder = SparsePilotEncoder(num_candidate_patterns=32, hidden_dim=32, num_layers=0, dropout=0.0).eval()
    observations = torch.complex(torch.randn(3, 2, 2), torch.randn(3, 2, 2))
    pattern_ids = torch.tensor([[0, 1]]).expand(3, -1)
    positions = torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ)
    mask = torch.ones(3, 2, 2, dtype=torch.bool)

    missing = encoder(observations, pattern_ids, positions, mask, None)
    measured = encoder(observations, pattern_ids, positions, mask, 10.0)

    torch.testing.assert_close(missing["csi_feature"], measured["csi_feature"])
    assert not bool(missing["snr_available"].any())
    assert bool(measured["snr_available"].all())
    assert torch.isfinite(missing["csi_quality"]).all()


def test_sparse_pilot_encoder_uses_complex_imaginary_evidence() -> None:
    torch.manual_seed(3)
    encoder = SparsePilotEncoder(num_candidate_patterns=32, hidden_dim=32, num_layers=0, dropout=0.0).eval()
    real = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    first = torch.complex(real, torch.zeros_like(real))
    second = torch.complex(real, torch.tensor([[[0.5, -1.0], [1.5, -2.0]]]))
    pattern_ids = torch.tensor([[0, 1]])
    positions = torch.tensor(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    first_feature = encoder(first, pattern_ids, positions, mask)["csi_feature"]
    second_feature = encoder(second, pattern_ids, positions, mask)["csi_feature"]

    assert not torch.allclose(first_feature, second_feature)


def test_fixed_sparse_csi_sidecar_matches_direct_complex_projection(tmp_path: Path) -> None:
    codebook = generate_probe_codebook(64, 16, num_patterns=32, seed=2026, method="random_qpsk")
    codebook_path = codebook.save(tmp_path / "codebook.npz")
    codebook_sha256 = hashlib.sha256(codebook_path.read_bytes()).hexdigest()
    rng = np.random.default_rng(4)
    paths = 3
    matrices = (
        rng.standard_normal((1, 1, 16, 1, 64, paths, 1))
        + 1j * rng.standard_normal((1, 1, 16, 1, 64, paths, 1))
    ).astype(np.complex64)
    delays = np.linspace(0.0, 3e-8, paths, dtype=np.float32).reshape(1, 1, 1, paths)
    channel_path = tmp_path / "000001.npz"
    np.savez(channel_path, a=matrices, tau=delays)
    sidecar = PCPFSparseCSISidecar(
        {
            "enabled": True,
            "codebook_path": str(codebook_path),
            "codebook_sha256": codebook_sha256,
            "codebook_hash": codebook.hash,
            "cache_root": str(tmp_path / "cache"),
            "selection_sha256": PCPF_SPARSE_CSI_SELECTION_SHA256,
        }
    )

    output = sidecar.load_history([channel_path] * 5, history_frame_ids=range(1, 6))
    a, tau = load_path_channel(channel_path)
    direct = simulate_candidate_pilots(
        a[None, None, :, None, :, :, None],
        tau[None, None, None, :],
        codebook,
        np.asarray(PCPF_SPARSE_CSI_FREQUENCY_POSITIONS_HZ),
    )[:2]

    assert output["csi"].shape == (5, 2, 2)
    assert torch.is_complex(output["csi"])
    torch.testing.assert_close(output["csi"][0], torch.from_numpy(direct))
    assert not bool(output["csi_snr_available"])
    assert PCPF_SPARSE_CSI_SELECTION_SHA256 == "87bad2292ba3d22cac413e71d9303f2dd229ed64fe39eeb4df6272f42e6bca28"
    assert sidecar.identity["sampling_ratio"] == pytest.approx(0.0078125)
    assert sidecar.identity["spatial_selection"]["direct_antenna_indices"] is None
    assert sidecar.identity["spatial_selection"]["tx_pattern_indices"] == [0, 1]
    assert sidecar.identity["spatial_selection"]["rx_pattern_indices"] == [0, 1]
    assert sidecar.identity["awgn_enabled"] is False
    assert sidecar.identity["pilot_dropout_enabled"] is False
    assert sidecar.identity["synthetic_corruption_enabled"] is False

    with pytest.raises(ValueError, match="consecutive and increasing"):
        sidecar.load_history([channel_path] * 5, history_frame_ids=[1, 2, 4, 5, 6])


def test_five_modality_schedule_cycles_all_31_subsets_equally() -> None:
    batch_size = 62
    batch = {
        "image": torch.ones(batch_size, 5, 1),
        "radar_ra": torch.ones(batch_size, 5, 1),
        "radar_da": torch.ones(batch_size, 5, 1),
        "gps": torch.ones(batch_size, 5, 1),
        "lidar": torch.ones(batch_size, 5, 1),
        "csi": torch.ones(batch_size, 5, 2, 2, dtype=torch.complex64),
        "csi_pilot_mask": torch.ones(batch_size, 5, 2, 2, dtype=torch.bool),
        "csi_valid_mask": torch.ones(batch_size, 5, dtype=torch.bool),
    }
    cfg = {
        "experiment": {"seed": 1},
        "model": {"primary": {"modalities": ["image", "radar", "gps", "lidar"], "use_sparse_csi": True}},
        "data": {"dataloader": {"train_batch_size": batch_size}},
        "temporal_missing": {
            "enabled": True,
            "mode": "balanced_pattern_schedule",
            "schedule_id": "pcpf_five_modality_all_subsets_v1",
            "history_window": 5,
            "panel_size": 31,
        },
    }

    result = apply_training_temporal_missing(batch, cfg, epoch=3, step=0)
    masks = result["modality_temporal_mask"]
    unique, counts = torch.unique(masks[:, 0], dim=0, return_counts=True)

    assert unique.shape == (31, 5)
    assert counts.tolist() == [2] * 31
    assert torch.equal(masks, masks[:, :1].expand_as(masks))
    assert len(set(result["temporal_missing_metadata"]["condition_ids"])) == 31
    assert "only_csi" in result["temporal_missing_metadata"]["condition_ids"]
    assert torch.count_nonzero(result["csi"][~masks[:, :, 4]]) == 0
    assert not bool(result["csi_pilot_mask"][~masks[:, :, 4]].any())


def test_default_four_modality_model_has_no_sparse_csi_state() -> None:
    model = PCPFTemporalRiskFusion(encoders=_encoders(), temporal_transformer={"dropout": 0.0})

    assert tuple(model.modalities) == ("image", "radar", "gps", "lidar")
    assert not any(key.startswith(("csi_encoder.", "csi_projection.")) for key in model.state_dict())


def test_sparse_csi_quality_projection_is_diagnostic_only() -> None:
    model = PCPFTemporalRiskFusion(
        encoders=_encoders(),
        use_sparse_csi=True,
        sparse_csi_encoder={"hidden_dim": 32, "num_layers": 0, "dropout": 0.0},
        training_stage="stage1_expert",
        temporal_transformer={"dropout": 0.0},
    )

    assert model.csi_encoder is not None
    assert all(not parameter.requires_grad for parameter in model.csi_encoder.quality_projection.parameters())
    assert any(parameter.requires_grad for parameter in model.csi_encoder.token_projection.parameters())


def test_csi_only_mask_has_unit_weight_and_one_shared_temporal_core() -> None:
    model = PCPFTemporalRiskFusion(
        encoders=_encoders(),
        use_sparse_csi=True,
        sparse_csi_encoder={"hidden_dim": 32, "num_layers": 0, "dropout": 0.0},
        training_stage="stage1_expert",
        fusion_mode="uniform",
        temporal_transformer={"dropout": 0.0},
    ).eval()
    mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    mask[:, :, 4] = True

    output = model(**_sensing_inputs(2), **_csi_inputs(2), modality_temporal_mask=mask)

    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, -1)
    torch.testing.assert_close(output["fusion_weights"], expected)
    assert torch.equal(output["available_modalities"], expected.bool())
    assert output["temporal_cls_features"].shape == (2, 5, 64)
    assert output["temporal_token_features"].shape == (2, 5, 5, 64)
    assert sum(isinstance(module, SharedTemporalTransformer) for module in model.modules()) == 1

    invalid_snr = _csi_inputs(2)
    invalid_snr["csi_snr_available"] = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="requires real csi_snr_db"):
        model(**_sensing_inputs(2), **invalid_snr, modality_temporal_mask=mask)


def test_epoch_mask_diagnostics_are_counts_not_batch_means() -> None:
    accumulator = EpochDiagnosticsAccumulator()
    accumulator.update({"loss/example": 2.0, "temporal_missing/mask_count/full": 3.0})
    accumulator.update({"loss/example": 4.0, "temporal_missing/mask_count/full": 2.0})

    summary = accumulator.mean()
    assert summary["loss/example"] == 3.0
    assert summary["temporal_missing/mask_count/full"] == 5.0
