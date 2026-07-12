from pathlib import Path

import torch

from kd_sensing.data.datasets.mmw import MMWDataset
from kd_sensing.data.sample_cache import LmdbSampleCache
from kd_sensing.preprocessing.sample_cache import (
    generate_deepsense6g_sample_lmdb_cache,
    generate_sample_lmdb_cache,
)


class _FakeDataset:
    def __init__(self, split: str):
        self.split = split
        self.root_csv = Path(f"{split}.csv")
        self.data_root = Path("dataset/MMW/rainy")
        self.enabled_modalities = ("image", "gps", "lidar")
        self.seq_len = 5
        self.num_pred = 1
        self.condition = "rainy"
        self.scene_slug = "Town03_5wayroad_seed28"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return {"sample_id": f"{self.split}:{idx}", "target_beam": torch.tensor([3])}


def test_sample_lmdb_cache_builds_registered_mmw_dataset_and_records_h5p1_metadata(tmp_path, monkeypatch):
    built = []

    def build(cfg):
        built.append(dict(cfg))
        return _FakeDataset(str(cfg["split"]))

    monkeypatch.setattr("kd_sensing.preprocessing.sample_cache.DATASETS.build", build)
    path = tmp_path / "mmw_{split}.lmdb"

    result = generate_sample_lmdb_cache(
        dataset={"type": "mmw", "condition": "rainy", "scene": "Town03_5wayroad_seed28"},
        path=str(path),
        progress=False,
        map_size_gb=0.01,
    )

    assert result["type"] == "sample_lmdb_cache"
    assert [cfg["type"] for cfg in built] == ["mmw", "mmw"]
    assert all(cfg["sample_cache"] is None for cfg in built)
    cache = LmdbSampleCache(tmp_path / "mmw_train.lmdb")
    metadata = cache.get("__metadata__")
    cache.close()
    assert metadata["dataset_type"] == "mmw"
    assert metadata["condition"] == "rainy"
    assert metadata["scenario"] == "Town03_5wayroad_seed28"
    assert metadata["seq_len"] == 5
    assert metadata["num_pred"] == 1


def test_legacy_deepsense6g_lmdb_generator_keeps_legacy_type(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )

    result = generate_deepsense6g_sample_lmdb_cache(
        dataset={"type": "deepsense6g"},
        path=str(tmp_path / "legacy_{split}.lmdb"),
        splits=["train"],
        progress=False,
        map_size_gb=0.01,
    )

    assert result["type"] == "deepsense6g_sample_lmdb_cache"


def test_mmw_lmdb_cache_hit_returns_complete_sample_without_reaugmenting():
    expected = {"target_beam": torch.tensor([7]), "metadata": {"condition": "foggy"}}
    dataset = object.__new__(MMWDataset)
    dataset.sample_cache = type("Cache", (), {"get": lambda self, key: expected})()
    dataset.family_adapter = type(
        "Adapter",
        (),
        {"augment_sample": lambda self, idx, sample: (_ for _ in ()).throw(AssertionError("reaugmented"))},
    )()
    dataset.split = "train"

    assert dataset[0] is expected
