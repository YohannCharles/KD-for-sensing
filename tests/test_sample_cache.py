from pathlib import Path

import pytest
import torch

from kd_sensing.data.sample_cache import LmdbSampleCache, sample_cache_path_for_split


def test_sample_cache_path_formats_split(tmp_path: Path):
    path = sample_cache_path_for_split(tmp_path / "sample_{split}.lmdb", "train")
    assert path.name == "sample_train.lmdb"


def test_lmdb_sample_cache_roundtrip(tmp_path: Path):
    pytest.importorskip("lmdb")
    cache = LmdbSampleCache(tmp_path / "samples.lmdb", readonly=False, map_size_gb=0.01)
    try:
        cache.put("train:0", {"image": torch.ones(1, 2), "label": torch.tensor([3])})
        cache.close()
        reader = LmdbSampleCache(tmp_path / "samples.lmdb", readonly=True)
        try:
            sample = reader.get("train:0")
        finally:
            reader.close()
    finally:
        cache.close()

    assert sample["image"].shape == (1, 2)
    assert int(sample["label"][0]) == 3
