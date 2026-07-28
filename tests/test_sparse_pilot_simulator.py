import numpy as np
import pytest

from kd_sensing.channel.probe_codebook import generate_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import (
    add_awgn,
    explicit_frequency_channel,
    frequency_offsets_hz,
    parse_path_channel,
    pilot_subcarrier_indices,
    project_explicit_channel,
    simulate_candidate_pilots,
)


def _channel(nr=4, nt=8, paths=3, seed=5):
    rng = np.random.default_rng(seed)
    matrices = (rng.normal(size=(nr, nt, paths)) + 1j * rng.normal(size=(nr, nt, paths))).astype(np.complex64)
    a = matrices[None, None, :, None, :, :, None]
    tau = np.linspace(0.0, 90e-9, paths, dtype=np.float32)[None, None, None, :]
    return a, tau


@pytest.mark.parametrize("nr,nt", [(2, 4), (4, 8), (16, 64)])
def test_path_domain_matches_explicit_frequency_channel(nr, nt):
    a, tau = _channel(nr=nr, nt=nt)
    codebook = generate_probe_codebook(nt, nr, num_patterns=4, seed=7)
    indices = pilot_subcarrier_indices(1024, 8)
    frequencies = frequency_offsets_hz(indices, num_subcarriers=1024, subcarrier_spacing_hz=120_000, mode="centered")

    direct = simulate_candidate_pilots(a, tau, codebook, frequencies)
    explicit = project_explicit_channel(explicit_frequency_channel(a, tau, frequencies), codebook)
    relative_error = np.linalg.norm(direct - explicit) / np.linalg.norm(explicit)
    assert relative_error < 1e-5


def test_npz_shape_parser_and_frequency_auto_fail_closed():
    a, tau = _channel()
    matrices, delays = parse_path_channel(a, tau)
    assert matrices.shape == (4, 8, 3)
    assert delays.shape == (3,)
    with pytest.raises(ValueError, match="auto"):
        frequency_offsets_hz(np.arange(4), num_subcarriers=8, subcarrier_spacing_hz=1.0, mode="auto")
    with pytest.raises(ValueError, match="shape"):
        parse_path_channel(a.squeeze(0), tau)


def test_awgn_matches_requested_snr():
    signal = np.ones((4096,), dtype=np.complex64) * (1.0 + 1.0j)
    noisy, expected_variance = add_awgn(signal, 10.0, rng=np.random.default_rng(11))
    measured_noise_power = float(np.mean(np.abs(noisy - signal) ** 2))
    measured_snr = 10.0 * np.log10(float(np.mean(np.abs(signal) ** 2)) / measured_noise_power)
    assert expected_variance == pytest.approx(0.2)
    assert measured_snr == pytest.approx(10.0, abs=0.35)
