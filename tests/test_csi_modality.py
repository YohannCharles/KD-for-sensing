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
    CSIDegradationConfig,
    CSIRMSNormalizer,
    degrade_csi_payload,
    load_csi_sequence,
    read_csi_tensor,
    resolve_csi_degradation_config,
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


def test_csi_degradation_profile_resolver_defaults_and_overrides():
    disabled = resolve_csi_degradation_config(None)
    medium = resolve_csi_degradation_config(
        {
            "enabled": True,
            "profile": "medium",
            "snr_db": 12.0,
            "temporal_shift_choices": [0, 2],
            "seed": 123,
        }
    )
    hard = resolve_csi_degradation_config({"enabled": True, "profile": "hard"})

    assert disabled.enabled is False
    assert disabled.profile == "clean"
    assert medium.enabled is True
    assert medium.profile == "medium"
    assert medium.snr_db == pytest.approx(12.0)
    assert medium.path_dropout_rate == pytest.approx(0.2)
    assert medium.delay_noise_std_ns == pytest.approx(0.5)
    assert medium.angle_noise_std_deg == pytest.approx(3.0)
    assert medium.antenna_phase_error_std_deg == pytest.approx(10.0)
    assert medium.temporal_shift_choices == (0, 2)
    assert medium.seed == 123
    assert hard.snr_db == pytest.approx(5.0)
    assert hard.path_dropout_rate == pytest.approx(0.3)
    assert hard.dominant_path_attenuation == pytest.approx(0.5)
    assert hard.temporal_shift_choices == (-2, -1, 0, 1, 2)


def test_csi_degradation_awgn_variance_on_tensor_payload():
    clean = np.ones((8192, 1), dtype=np.complex64)
    diagnostics: dict[str, object] = {}

    degraded, _ = degrade_csi_payload(
        {"channel": clean},
        config=CSIDegradationConfig(enabled=True, profile="clean", snr_db=10.0),
        rng=np.random.default_rng(123),
        diagnostics=diagnostics,
    )
    noise = _complex_from_real_imag(degraded) - clean

    assert diagnostics["source_mode"] == "tensor"
    assert diagnostics["awgn"]["complex_noise_variance"] == pytest.approx(0.1)
    assert float(noise.real.var()) == pytest.approx(0.05, rel=0.2)
    assert float(noise.imag.var()) == pytest.approx(0.05, rel=0.2)


def test_csi_degradation_path_dropout_preserves_dominant_path():
    payload = {
        "path_gain": np.asarray([4.0, 2.0, 1.0, 0.5], dtype=np.complex64),
        "aod": np.zeros(4, dtype=np.float32),
    }
    diagnostics: dict[str, object] = {}

    degraded, _ = degrade_csi_payload(
        payload,
        config=CSIDegradationConfig(enabled=True, profile="clean", path_dropout_rate=0.5, tx_antennas=4),
        rng=np.random.default_rng(5),
        diagnostics=diagnostics,
    )

    assert degraded.shape == (1, 4, 2)
    assert diagnostics["source_mode"] == "path"
    assert diagnostics["path_dropout"]["dropped_indices"] == [3, 2]
    assert diagnostics["path_dropout"]["dominant_index"] == 0
    assert 0 not in diagnostics["path_dropout"]["dropped_indices"]


def test_csi_degradation_dominant_delay_angle_and_phase_operators():
    attenuation_payload = {
        "path_gain": np.asarray([4.0, 2.0], dtype=np.complex64),
        "aod": np.zeros(2, dtype=np.float32),
    }
    attenuated, _ = degrade_csi_payload(
        attenuation_payload,
        config=CSIDegradationConfig(enabled=True, profile="clean", dominant_path_attenuation=0.5, tx_antennas=4),
        rng=np.random.default_rng(1),
        diagnostics={},
    )
    np.testing.assert_allclose(attenuated[..., 0], 4.0, atol=1e-5)

    payload = {
        "path_gain": np.asarray([1.0, 0.5], dtype=np.complex64),
        "aod": np.asarray([0.0, 15.0], dtype=np.float32),
        "delay": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    clean, _ = degrade_csi_payload(
        payload,
        config=CSIDegradationConfig(enabled=True, profile="clean", tx_antennas=4),
        rng=np.random.default_rng(7),
        diagnostics={},
    )
    diagnostics: dict[str, object] = {}
    degraded, _ = degrade_csi_payload(
        payload,
        config=CSIDegradationConfig(
            enabled=True,
            profile="clean",
            delay_noise_std_ns=0.5,
            delay_quantization_ns=0.25,
            angle_noise_std_deg=3.0,
            antenna_phase_error_std_deg=10.0,
            tx_antennas=4,
        ),
        rng=np.random.default_rng(7),
        diagnostics=diagnostics,
    )

    assert not np.allclose(degraded, clean)
    assert "delay_noise" in diagnostics
    assert "delay_quantization" in diagnostics
    assert "angle_noise" in diagnostics
    assert "antenna_phase_error" in diagnostics
    assert diagnostics["skipped_operators"] == []


def test_csi_degradation_tensor_only_records_skipped_path_operators():
    diagnostics: dict[str, object] = {}

    degraded, _ = degrade_csi_payload(
        {"channel": np.ones((4, 2), dtype=np.complex64)},
        config=CSIDegradationConfig(
            enabled=True,
            profile="clean",
            path_dropout_rate=0.5,
            dominant_path_attenuation=0.5,
            delay_noise_std_ns=0.5,
            angle_noise_std_deg=3.0,
        ),
        rng=np.random.default_rng(11),
        diagnostics=diagnostics,
    )

    assert degraded.shape == (4, 2, 2)
    assert diagnostics["source_mode"] == "tensor"
    assert set(diagnostics["skipped_operators"]) == {
        "path_dropout",
        "dominant_path_attenuation",
        "delay_noise",
        "angle_noise",
    }


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


def test_degraded_csi_dataset_is_deterministic_and_rms_uses_clean(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_csi_sequence_fixture(tmp_path, train_csv, prefix="train", seq_len=3, num_pred=2, amplitude=2.0)
    _write_csi_sequence_fixture(tmp_path, test_csv, prefix="test", seq_len=3, num_pred=2, amplitude=4.0)
    degradation = {"enabled": True, "profile": "clean", "snr_db": 0.0, "seed": 99}

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=3,
        num_pred=2,
        enabled_modalities=["csi"],
        use_csi=True,
        csi_train_rms=True,
        csi_degradation=degradation,
    )
    repeat_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=3,
        num_pred=2,
        enabled_modalities=["csi"],
        use_csi=True,
        csi_train_rms=True,
        csi_degradation=degradation,
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
        csi_degradation=degradation,
    )

    first = train_dataset[0]["csi"].numpy()
    train_dataset._csi_degraded_cache.clear()
    train_dataset._csi_degradation_diagnostics.clear()
    second = train_dataset[0]["csi"].numpy()
    repeated = repeat_dataset[0]["csi"].numpy()

    assert train_dataset.csi_rms_normalizer.rms == pytest.approx(2.0)
    assert test_dataset.csi_rms_normalizer is train_dataset.csi_rms_normalizer
    assert not np.allclose(first[..., 0], 2.0)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, repeated)


def test_csi_temporal_shift_clamps_history_window_and_records_metadata(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    _write_csi_sequence_fixture(
        tmp_path,
        train_csv,
        prefix="shift",
        seq_len=3,
        num_pred=1,
        amplitude=1.0,
        amplitudes=[1.0, 2.0, 3.0],
    )

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=3,
        num_pred=1,
        enabled_modalities=["csi"],
        use_csi=True,
        csi_train_rms=False,
        csi_degradation={"enabled": True, "profile": "clean", "temporal_shift_choices": [1], "seed": 3},
        return_metadata=True,
    )
    sample = dataset[0]

    np.testing.assert_allclose(sample["csi"][:, 0, 0, 0].numpy(), [2.0, 3.0, 3.0])
    metadata = sample["metadata"]["csi_degradation"]
    assert metadata["temporal_shift"] == 1
    assert metadata["temporal_fill_mode"] == "clamp"
    assert metadata["skipped_operators"] == []
    assert metadata["resolved_parameters"]["profile"] == "clean"


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
    degraded_csi_cfg = load_config(ROOT / "configs/csi/medium_degraded_no_kd.yaml")
    degraded_fusion_cfg = load_config(ROOT / "configs/fusion/mmwave_csi_medium_degraded_no_kd.yaml")

    assert csi_cfg["experiment"]["task"] == "csi"
    assert csi_cfg["model"]["student"]["encoders"]["csi"]["type"] == "pilot_dual_view_csi"
    assert "csi_degradation" not in csi_cfg["data"]["dataset"]
    assert fusion_cfg["model"]["student"]["modalities"] == ["mmwave", "csi"]
    assert fusion_cfg["data"]["dataset"]["use_csi"] is True
    assert "csi_degradation" not in fusion_cfg["data"]["dataset"]
    assert degraded_csi_cfg["experiment"]["task"] == "csi"
    assert degraded_csi_cfg["data"]["dataset"]["use_csi"] is True
    assert degraded_csi_cfg["data"]["dataset"]["csi_degradation"]["enabled"] is True
    assert degraded_csi_cfg["data"]["dataset"]["csi_degradation"]["profile"] == "medium"
    assert degraded_fusion_cfg["model"]["student"]["modalities"] == ["mmwave", "csi"]
    assert degraded_fusion_cfg["data"]["dataset"]["csi_degradation"]["profile"] == "medium"

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
    amplitudes: list[float] | None = None,
) -> None:
    csi_paths = []
    beam_paths = []
    future_paths = []
    for idx in range(seq_len):
        csi_name = f"{prefix}_csi_{idx}.npy"
        beam_name = f"{prefix}_beam_{idx}.txt"
        csi_amplitude = float(amplitudes[idx]) if amplitudes is not None else float(amplitude)
        csi = np.full((4, 2), complex(csi_amplitude, 0.0), dtype=np.complex64)
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


def _complex_from_real_imag(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    return values[..., 0] + 1j * values[..., 1]


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
