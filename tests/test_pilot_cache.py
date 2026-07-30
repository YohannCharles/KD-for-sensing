import numpy as np

from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec


def test_pilot_cache_invalidates_when_spec_or_channel_changes(tmp_path):
    channel = tmp_path / "frame.npz"
    np.savez(channel, a=np.ones((1, 1, 2, 1, 4, 1, 1), dtype=np.complex64), tau=np.zeros((1, 1, 1, 1)))
    cache = PilotCache(tmp_path / "cache")
    base = PilotCacheSpec("codebook-a", (0.0, 1.0), 1.0, "centered", 4, 2)
    values = np.ones((3, 2), dtype=np.complex64)

    cache.store(channel, base, values)
    assert np.array_equal(cache.load(channel, base), values)
    changed = PilotCacheSpec("codebook-b", (0.0, 1.0), 1.0, "centered", 4, 2)
    assert cache.load(channel, changed) is None

    np.savez(channel, a=np.zeros((1, 1, 2, 1, 4, 1, 1), dtype=np.complex64), tau=np.zeros((1, 1, 1, 1)))
    assert cache.load(channel, base) is None


def test_pilot_cache_hashes_a_miss_only_once(tmp_path, monkeypatch):
    channel = tmp_path / "frame.npz"
    np.savez(channel, a=np.ones((1,), dtype=np.complex64), tau=np.zeros((1,)))
    cache = PilotCache(tmp_path / "cache")
    spec = PilotCacheSpec("codebook", (0.0, 1.0), 1.0, "centered", 4, 2)
    calls = 0

    from kd_sensing.channel import pilot_cache as pilot_cache_module

    original = pilot_cache_module._sha256_file

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(pilot_cache_module, "_sha256_file", counted)
    values, key = cache.get_or_compute_with_key(channel, spec, lambda: np.ones((3, 2), dtype=np.complex64))

    assert values.shape == (3, 2)
    assert len(key) == 64
    assert calls == 1
