import numpy as np

from kd_sensing.channel.probe_codebook import generate_probe_codebook, load_probe_codebook


def test_random_qpsk_codebook_is_constant_modulus_and_reproducible(tmp_path):
    first = generate_probe_codebook(64, 16, num_patterns=32, seed=17)
    second = generate_probe_codebook(64, 16, num_patterns=32, seed=17)
    other = generate_probe_codebook(64, 16, num_patterns=32, seed=18)

    assert np.array_equal(first.tx, second.tx)
    assert np.array_equal(first.rx, second.rx)
    assert first.hash == second.hash
    assert first.hash != other.hash
    assert np.allclose(np.linalg.norm(first.tx, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(first.rx, axis=1), 1.0)
    assert np.allclose(np.abs(first.tx), 1.0 / np.sqrt(64))
    assert np.allclose(np.abs(first.rx), 1.0 / np.sqrt(16))

    path = first.save(tmp_path / "pilot_codebook.npz")
    loaded = load_probe_codebook(path)
    assert loaded.hash == first.hash
    assert np.array_equal(loaded.tx, first.tx)
    assert np.array_equal(loaded.rx, first.rx)


def test_all_probe_codebook_baselines_obey_hardware_constraint():
    for method in ("random_qpsk", "fixed_dft_beams", "multice_interleaved"):
        codebook = generate_probe_codebook(8, 4, num_patterns=4, seed=3, method=method)
        assert np.allclose(np.abs(codebook.tx), 1.0 / np.sqrt(8))
        assert np.allclose(np.abs(codebook.rx), 1.0 / np.sqrt(4))
