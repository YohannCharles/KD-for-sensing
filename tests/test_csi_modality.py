from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.data.mmw.preparation import build_sequence_rows  # noqa: E402
from kd_sensing.data.transform_ops.csi import (  # noqa: E402
    CSIRMSNormalizer,
    load_csi_sequence,
    read_csi_tensor,
)
from kd_sensing.engine.batch import forward_model, prepare_csi_inputs, prepare_fusion_inputs, prepare_labels  # noqa: E402
from kd_sensing.models.csi import PilotCSIChannelEstimator, PilotDualViewCSIEncoder  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.registries import ENCODERS, import_default_components  # noqa: E402


def test_csi_loader_accepts_complex_and_real_imag_and_rejects_bad_values(tmp_path: Path):
    complex_frame = (np.ones((4, 2), dtype=np.float32) + 1j * np.full((4, 2), 2.0, dtype=np.float32)).astype(
        np.complex64
    )
    real_imag_frame = np.stack([np.ones((4, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32)], axis=-1)
    np.save(tmp_path / "complex.npy", complex_frame)
    np.savez(tmp_path / "realimag.npz", csi=real_imag_frame)
    bad = real_imag_frame.copy()
    bad[0, 0, 0] = np.nan
    np.save(tmp_path / "bad.npy", bad)

    complex_loaded = read_csi_tensor(tmp_path, "complex.npy")
    real_imag_loaded = read_csi_tensor(tmp_path, "realimag.npz")
    sequence = load_csi_sequence(tmp_path, ["complex.npy", "realimag.npz"], seq_len=2)

    assert complex_loaded.shape == (4, 2, 2)
    assert complex_loaded.dtype == np.float32
    np.testing.assert_allclose(complex_loaded[..., 1], 2.0)
    assert real_imag_loaded.shape == (4, 2, 2)
    assert sequence.shape == (2, 4, 2, 2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        read_csi_tensor(tmp_path, "bad.npy")


def test_csi_dataset_fits_rms_on_train_and_reuses_for_test(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_csi_sequence_fixture(tmp_path, train_csv, prefix="train", seq_len=3, num_pred=2, amplitude=2.0)
    _write_csi_sequence_fixture(tmp_path, test_csv, prefix="test", seq_len=3, num_pred=2, amplitude=4.0)

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=3,
        num_pred=2,
        enabled_modalities=["csi"],
        use_csi=True,
        csi_train_rms=True,
    )
    test_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(test_csv),
        split="test",
        seq_len=3,
        num_pred=2,
        enabled_modalities=["csi"],
        use_csi=True,
        csi_train_rms=True,
        csi_rms_normalizer=train_dataset.csi_rms_normalizer,
    )

    assert isinstance(train_dataset.csi_rms_normalizer, CSIRMSNormalizer)
    assert train_dataset.csi_rms_normalizer.rms == pytest.approx(2.0)
    assert test_dataset.csi_rms_normalizer is train_dataset.csi_rms_normalizer
    assert train_dataset[0]["csi"].shape == (3, 4, 2, 2)
    assert float(train_dataset[0]["csi"][0, 0, 0, 0]) == pytest.approx(2.0)
    assert float(test_dataset[0]["csi"][0, 0, 0, 0]) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="train-fitted csi_rms_normalizer"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(test_csv),
            split="test",
            seq_len=3,
            num_pred=2,
            enabled_modalities=["csi"],
            use_csi=True,
            csi_train_rms=True,
        )


def test_csi_batch_padding_labels_and_modular_forward_contract():
    batch = {
        "csi": torch.ones(2, 4, 5, 3, 2),
        "target_beam": torch.tensor([[7, 8, 9], [10, 11, 12]]),
    }
    csi_input = prepare_csi_inputs(batch, seq_length=3, num_pred=3, device=torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(batch, seq_length=3, num_pred=3, device=torch.device("cpu"), modalities=["csi"])
    labels = prepare_labels(batch, num_pred=3, downsample_ratio=1, device=torch.device("cpu"))

    model = ModularSequenceModel(
        modalities=["csi"],
        feature_size=16,
        d_model=16,
        num_classes=64,
        num_pred=3,
        encoders={"csi": {"type": "pilot_dual_view_csi", "output_dim": 16, "delay_taps": 4}},
        representation_core={"type": "single_gru", "d_model": 16, "hidden_size": 16},
    )
    with torch.no_grad():
        output = forward_model(model, "csi", csi_batch=csi_input)

    assert csi_input.shape == (2, 5, 5, 3, 2)
    assert torch.count_nonzero(csi_input[:, -2:]) == 0
    assert sorted(fusion_inputs) == ["csi_batch"]
    assert labels.tolist() == [[7, 8, 9], [10, 11, 12]]
    assert output["logits"].shape == (2, 5, 64)


def test_pilot_estimator_noise_variance_and_encoder_registry_shape():
    clean = torch.ones(512, 2, 4, 2, dtype=torch.complex64)
    estimator = PilotCSIChannelEstimator(mode="physical", pilot_len=16, pilot_power=1.0, noise_var=0.01)
    estimator.train()

    noisy, aux = estimator(clean, return_aux=True)
    noise = noisy - clean
    real_var = float(noise.real.var(unbiased=False))
    imag_var = float(noise.imag.var(unbiased=False))

    assert float(aux["sigma_e2"]) == pytest.approx(0.01 / 16.0)
    assert real_var == pytest.approx((0.01 / 16.0) / 2.0, rel=0.25)
    assert imag_var == pytest.approx((0.01 / 16.0) / 2.0, rel=0.25)

    import_default_components()
    encoder = ENCODERS.build(
        {
            "type": "pilot_dual_view_csi",
            "output_dim": 32,
            "train_rms": 1.0,
            "csi_estimation": {"mode": "est_snr", "snr_db": 20.0, "pilot_len": 8, "pilot_power": 1.0},
            "delay_taps": 4,
            "view_fusion": "symmetric_gate",
            "return_aux": True,
        }
    )
    assert isinstance(encoder, PilotDualViewCSIEncoder)
    real_imag = torch.randn(2, 5, 8, 4, 2)
    with torch.no_grad():
        encoded, aux = encoder(real_imag)
        complex_encoded = encoder(torch.complex(real_imag[..., 0], real_imag[..., 1]), return_aux=False)
    assert encoded.shape == (2, 5, 32)
    assert complex_encoded.shape == (2, 5, 32)
    assert aux["view_gate"].shape == (2, 5, 2)


def test_csi_configs_load_and_mmw_sequences_include_csi_columns():
    csi_cfg = load_config(ROOT / "configs/csi/no_kd.yaml")
    fusion_cfg = load_config(ROOT / "configs/fusion/mmwave_csi_no_kd.yaml")

    assert csi_cfg["experiment"]["task"] == "csi"
    assert csi_cfg["model"]["student"]["encoders"]["csi"]["type"] == "pilot_dual_view_csi"
    assert fusion_cfg["model"]["student"]["modalities"] == ["mmwave", "csi"]
    assert fusion_cfg["data"]["dataset"]["use_csi"] is True

    rows, _ = build_sequence_rows(_prepared_frames(12), seq_len=8, pred_len=3)
    assert rows[0]["csi1"].endswith("000000_paths.npy")
    assert rows[0]["csi8"].endswith("000007_paths.npy")


def _write_csi_sequence_fixture(
    root: Path,
    csv_path: Path,
    *,
    prefix: str,
    seq_len: int,
    num_pred: int,
    amplitude: float,
) -> None:
    csi_paths = []
    beam_paths = []
    future_paths = []
    for idx in range(seq_len):
        csi_name = f"{prefix}_csi_{idx}.npy"
        beam_name = f"{prefix}_beam_{idx}.txt"
        csi = np.full((4, 2), complex(amplitude, 0.0), dtype=np.complex64)
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.save(root / csi_name, csi)
        np.savetxt(root / beam_name, beam)
        csi_paths.append(csi_name)
        beam_paths.append(beam_name)
    for idx in range(num_pred):
        future_name = f"{prefix}_future_{idx}.txt"
        future = np.zeros(64, dtype=np.float32)
        future[idx + 8] = 1.0
        np.savetxt(root / future_name, future)
        future_paths.append(future_name)
    columns = (
        [f"csi{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    values = csi_paths + beam_paths + future_paths + ["1"]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _prepared_frames(count: int):
    from kd_sensing.data.mmw.preparation import PreparedFrame

    frames = []
    for idx in range(count):
        frame_id = f"{idx:06d}"
        frames.append(
            PreparedFrame(
                agent="cav_0",
                frame_id=frame_id,
                camera0=f"Sensor_Data/cav_0/{frame_id}_camera0.png",
                cameras={"camera0": f"Sensor_Data/cav_0/{frame_id}_camera0.png"},
                lidar=f"Sensor_Data/cav_0/{frame_id}.pcd",
                gps=f"Sensor_Data/cav_0/{frame_id}.yaml",
                channel_path=f"Channel_Data/cav_0/{frame_id}_paths.npy",
                beam_power_path=f"Prepared/beam_power/cav_0/{frame_id}.txt",
                beam_label=0,
                rsu={},
            )
        )
    return frames
