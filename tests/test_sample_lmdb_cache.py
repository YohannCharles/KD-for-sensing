import json
from pathlib import Path

import pytest
import torch

from kd_sensing.data.datasets.mmw import MMWDataset
from kd_sensing.data.sample_cache import LmdbSampleCache
from kd_sensing.preprocessing.sample_cache import (
    SAMPLE_CACHE_MARKER,
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
        cache_root=str(tmp_path),
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
    assert (tmp_path / "mmw_train.lmdb" / SAMPLE_CACHE_MARKER).is_file()


def test_legacy_deepsense6g_lmdb_generator_keeps_legacy_type(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )

    result = generate_deepsense6g_sample_lmdb_cache(
        dataset={"type": "deepsense6g"},
        path=str(tmp_path / "legacy_{split}.lmdb"),
        cache_root=str(tmp_path),
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


def test_sample_cache_overwrite_requires_matching_owner_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )
    path = tmp_path / "cache_{split}.lmdb"
    target = tmp_path / "cache_train.lmdb"
    target.mkdir()
    sentinel = target / "do-not-delete.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="unowned sample cache"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(path),
            cache_root=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
            map_size_gb=0.01,
        )

    assert sentinel.exists()


def test_sample_cache_overwrite_rejects_mismatched_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )
    target = tmp_path / "cache_train.lmdb"
    target.mkdir()
    (target / SAMPLE_CACHE_MARKER).write_text(
        json.dumps({"schema_version": 1, "owner": "other", "path": str(target)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mismatched ownership"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path / "cache_{split}.lmdb"),
            cache_root=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
            map_size_gb=0.01,
        )

    assert target.exists()


def test_sample_cache_overwrite_rejects_marker_without_lmdb_files(tmp_path):
    target = tmp_path / "cache_train.lmdb"
    target.mkdir()
    marker = target / SAMPLE_CACHE_MARKER
    marker.write_text(
        json.dumps({"schema_version": 1, "owner": "sample_lmdb_cache", "path": str(target.resolve())}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid LMDB structure"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path / "cache_{split}.lmdb"),
            cache_root=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
        )

    assert marker.is_file()


def test_sample_cache_overwrite_rejects_symlinked_lmdb_file(tmp_path):
    target = tmp_path / "cache_train.lmdb"
    target.mkdir()
    (target / SAMPLE_CACHE_MARKER).write_text(
        json.dumps({"schema_version": 1, "owner": "sample_lmdb_cache", "path": str(target.resolve())}),
        encoding="utf-8",
    )
    (target / "data.mdb").write_bytes(b"not-an-lmdb")
    outside_lock = tmp_path / "outside-lock.mdb"
    outside_lock.write_bytes(b"keep")
    (target / "lock.mdb").symlink_to(outside_lock)

    with pytest.raises(ValueError, match="invalid LMDB structure"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path / "cache_{split}.lmdb"),
            cache_root=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
        )

    assert (target / "lock.mdb").is_symlink()
    assert outside_lock.read_bytes() == b"keep"


def test_sample_cache_overwrite_rejects_project_and_dataset_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("KD_SENSING_ROOT", str(tmp_path))
    (tmp_path / "dataset").mkdir()

    with pytest.raises(ValueError, match="Unsafe sample cache path"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
        )
    with pytest.raises(ValueError, match="Unsafe sample cache path"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path / "dataset"),
            splits=["train"],
            overwrite=True,
            progress=False,
        )


def test_sample_cache_overwrite_rejects_symlink_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "cache_train.lmdb"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(tmp_path / "cache_{split}.lmdb"),
            cache_root=str(tmp_path),
            splits=["train"],
            overwrite=True,
            progress=False,
        )

    assert outside.exists()


def test_sample_cache_default_root_rejects_arbitrary_external_path(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external"
    monkeypatch.setenv("KD_SENSING_ROOT", str(project))

    with pytest.raises(ValueError, match="Unsafe sample cache path"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(external / "cache_{split}.lmdb"),
            splits=["train"],
            overwrite=True,
            progress=False,
        )

    assert not external.exists()


def test_sample_cache_rejects_symlinked_parent_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.preprocessing.sample_cache.DATASETS.build",
        lambda cfg: _FakeDataset(str(cfg["split"])),
    )
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        generate_sample_lmdb_cache(
            dataset={"type": "mmw"},
            path=str(allowed / "linked" / "cache_{split}.lmdb"),
            cache_root=str(allowed),
            splits=["train"],
            overwrite=True,
            progress=False,
        )

    assert not (outside / "cache_train.lmdb").exists()
