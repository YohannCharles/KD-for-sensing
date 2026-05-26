from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles  # noqa: E402
import kd_sensing.data.datasets.multimodal_nf as multimodal_nf_module  # noqa: E402
import kd_sensing.preprocessing.multimodal_nf_derived_cache as derived_cache_module  # noqa: E402
from kd_sensing.data.datasets.multimodal_nf import MultimodalNFDataset  # noqa: E402
from kd_sensing.data.dataset_runtime import RuntimeDataset, SampleIndex, SampleRow  # noqa: E402
from kd_sensing.engine.batch import prepare_csi_inputs, prepare_lidar_inputs  # noqa: E402
from kd_sensing.engine.data_factory import build_dataset  # noqa: E402
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata  # noqa: E402
from kd_sensing.preprocessing.multimodal_nf_common import (  # noqa: E402
    audit_multimodal_nf_files,
    build_multimodal_nf_index,
    flatten_beam_triplet,
    parse_codebook_metadata,
    unflatten_beam_class,
)
from kd_sensing.preprocessing.multimodal_nf_derived_cache import (  # noqa: E402
    build_expected_metadata,
    cache_status,
    prewarm_multimodal_nf_derived_cache,
    sidecar_path,
)


CODEBOOK_SHAPE = (2, 3, 4)


def test_multimodal_nf_audit_fixture_and_codebook_metadata(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path)
    report = audit_multimodal_nf_files(
        data_root=tmp_path,
        channel_path=channel_path,
        codebook_path=codebook_path,
        output_dir=tmp_path / "audit",
        require_complete=True,
    )

    assert report["sample_count"] == 6
    assert report["missing_fields"] == []
    assert {"City_A", "City_B", "City_C"} <= set(report["city_ids"])
    assert "H" in report["hdf5_files"][0]["datasets"]
    assert report["hdf5_files"][0]["datasets"]["H"]["shape"] == [6, 4, 2, 2]

    dense = parse_codebook_metadata(profile="dense")
    small = parse_codebook_metadata(profile="small")
    parsed = parse_codebook_metadata(codebook_path)
    assert dense["shape"] == [90, 45, 16]
    assert small["shape"] == [20, 20, 10]
    assert parsed["shape"] == list(CODEBOOK_SHAPE)
    class_id = flatten_beam_triplet([1, 2, 3], CODEBOOK_SHAPE)
    assert class_id == 23
    assert unflatten_beam_class(class_id, CODEBOOK_SHAPE) == (1, 2, 3)

    bad_path = tmp_path / "raw" / "bad.h5"
    _write_fixture_h5(bad_path, include_beam_power=False)
    with pytest.raises(ValueError, match="missing required"):
        audit_multimodal_nf_files(
            data_root=tmp_path,
            channel_path=bad_path,
            codebook_path=codebook_path,
            output_dir=tmp_path / "bad_audit",
            require_complete=True,
        )


def test_multimodal_nf_descriptor_query_is_lightweight():
    descriptor = dataset_descriptor("multimodal_nf")
    assert descriptor.storage_kind == "hdf5_frame"
    assert descriptor.default_root == "dataset/MultimodalNF"
    assert descriptor.profile_for("csi").default_profile == "xl_mimo_nf"
    assert dataset_descriptor("deepsense6g").storage_kind == "csv_sequence"
    assert dataset_descriptor("raymobtime_s008").storage_kind == "npz_snapshot"

    code = f"""
import json
import sys
sys.path.insert(0, {str(SRC)!r})
from kd_sensing.data.dataset_descriptors import dataset_descriptor
dataset_descriptor("multimodal_nf")
print(json.dumps({{
    "torch": "torch" in sys.modules,
    "h5py": "h5py" in sys.modules,
    "pandas": "pandas" in sys.modules,
    "models": any(name.startswith("kd_sensing.models") for name in sys.modules),
}}, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == {"h5py": False, "models": False, "pandas": False, "torch": False}


def test_multimodal_nf_dataset_outputs_flat_fields_and_metadata(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_path=str(codebook_path),
        split_mode="city",
        split="train",
        enabled_modalities=["image", "lidar", "gps", "csi"],
        csi_subcarrier_policy="single",
        seq_len=3,
        num_pred=2,
        return_metadata=True,
    )

    sample = dataset[0]
    assert set(
        ["image", "lidar", "gps", "csi", "target_beam", "beam_triplet_topk", "beam_power_topk", "link_quality"]
    ).issubset(sample)
    assert sample["image"].shape == (3, 3, 5, 6)
    assert sample["lidar"].shape == (3, 10, 3)
    assert sample["gps"].shape == (3, 3)
    assert sample["csi"].shape == (3, 4, 1, 2)
    assert sample["target_beam"].shape == (2,)
    assert sample["beam_triplet_topk"].shape == (2, 5, 3)
    assert sample["beam_power_topk"].shape == (2, 5)
    assert sample["link_quality"].shape == (2,)
    assert sample["metadata"]["history_frame_ids"] == ["0", "1", "2"]
    assert sample["metadata"]["target_frame_ids"] == ["3", "4"]
    assert sample["metadata"]["dataset_type"] == "multimodal_nf"
    assert sample["metadata"]["input_profiles"]["lidar"] == "point_cloud_xyz_10000"
    assert sample["metadata"]["codebook"]["shape"] == list(CODEBOOK_SHAPE)
    assert dataset.auxiliary_target_metadata()["beam"]["num_beam_classes"] == 24


def test_multimodal_nf_disabled_modalities_do_not_require_missing_files(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        image_path=str(tmp_path / "missing_image.h5"),
        lidar_path=str(tmp_path / "missing_lidar.h5"),
        codebook_shape=list(CODEBOOK_SHAPE),
        split_mode="city",
        split="train",
        enabled_modalities=["gps", "csi"],
        seq_len=3,
        num_pred=2,
        return_metadata=False,
    )

    sample = dataset[0]
    assert set(sample) >= {"gps", "csi", "target_beam"}
    assert "image" not in sample
    assert "lidar" not in sample


def test_multimodal_nf_default_metadata_off_and_explicit_metadata_collates_mixed_aux_fields(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir(parents=True, exist_ok=True)
    codebook_path = codebooks / "small_codebook.pkl"
    with codebook_path.open("wb") as handle:
        pickle.dump({"shape": CODEBOOK_SHAPE, "name": "fixture_small"}, handle)
    _write_fixture_h5(
        raw / "City_A_dataset.h5",
        n=6,
        city_values=[b"City_A"] * 6,
        trajectory_values=[0] * 6,
        frame_values=list(range(6)),
        include_traj_nlos=True,
        include_mode=True,
    )
    _write_fixture_h5(
        raw / "City_B_dataset.h5",
        n=6,
        city_values=[b"City_B"] * 6,
        trajectory_values=[0] * 6,
        frame_values=list(range(6)),
        include_traj_nlos=False,
        include_mode=False,
    )
    common = {
        "data_root": str(tmp_path),
        "codebook_path": str(codebook_path),
        "split_mode": "city",
        "train_cities": ["City_A", "City_B"],
        "val_cities": [],
        "test_cities": [],
        "split": "train",
        "enabled_modalities": ["gps"],
        "seq_len": 3,
        "num_pred": 2,
    }

    training_dataset = MultimodalNFDataset(**common)
    assert "metadata" not in training_dataset[0]

    metadata_dataset = MultimodalNFDataset(**common, return_metadata=True)
    batch = next(iter(DataLoader(metadata_dataset, batch_size=len(metadata_dataset), num_workers=0)))
    assert "traj_nlos" in batch["metadata"]["resource_refs"]["hdf5_keys"]
    assert "" in batch["metadata"]["resource_refs"]["hdf5_keys"]["traj_nlos"]
    assert "mode" in batch["metadata"]["resource_refs"]["hdf5_keys"]


def test_multimodal_nf_shape_errors_include_context(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, bad_lidar=True, sequence=True)
    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_shape=list(CODEBOOK_SHAPE),
        split_mode="city",
        split="train",
        enabled_modalities=["lidar"],
        seq_len=3,
        num_pred=2,
    )

    with pytest.raises(ValueError, match="family=MultimodalNF modality=lidar profile=point_cloud_xyz_10000"):
        dataset[0]


def test_multimodal_nf_adapter_caches_hdf5_dataset_key_resolution(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "City_A_img.h5"
    _write_fixture_h5(image_path, n=4)
    row = SampleRow(
        sample_id="sample",
        split="train",
        dataset_type="multimodal_nf",
        family="MultimodalNF",
        resource_refs={
            "image_path": str(image_path),
            "channel_index": 1,
            "history_indices": [0, 1],
            "hdf5_keys": {},
        },
    )
    calls = 0
    original = multimodal_nf_module._dataset_paths

    def counted_dataset_paths(handle):
        nonlocal calls
        calls += 1
        return original(handle)

    monkeypatch.setattr(multimodal_nf_module, "_dataset_paths", counted_dataset_paths)
    adapter = multimodal_nf_module._MultimodalNFAdapter(
        modality="image",
        profile="rgb_imagenet",
        sample_key="image",
        csi_subcarrier_policy="all",
        csi_subcarrier_index=None,
    )

    try:
        first = adapter.load(row)
        second = adapter.load(row)
    finally:
        adapter.close()

    assert first["image"].shape == second["image"].shape == (2, 3, 5, 6)
    assert calls == 1


def test_multimodal_nf_eval_portion_does_not_truncate_train_split(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    channel_path = raw / "City_fixture.h5"
    _write_fixture_h5(
        channel_path,
        n=20,
        city_values=[b"City_A"] * 20,
        trajectory_values=[0] * 20,
        frame_values=list(range(20)),
    )

    common = {
        "data_root": str(tmp_path),
        "channel_path": str(channel_path),
        "codebook_shape": list(CODEBOOK_SHAPE),
        "split_mode": "frame_debug",
        "split_ratios": (0.5, 0.0, 0.5),
        "enabled_modalities": ["gps"],
        "seq_len": 3,
        "num_pred": 2,
        "eval_portion": 0.5,
    }
    train_dataset = MultimodalNFDataset(split="train", **common)
    test_dataset = MultimodalNFDataset(split="test", **common)

    assert len(train_dataset) == 8
    assert len(test_dataset) == 3
    assert train_dataset.sample_index.metadata["selected_portion"] == 1.0
    assert test_dataset.sample_index.metadata["selected_portion"] == 0.5


def test_multimodal_nf_hdf5_reader_uses_slice_for_contiguous_windows(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    h5py = pytest.importorskip("h5py")
    with h5py.File(channel_path, "r") as handle:
        sliced = multimodal_nf_module._read_hdf5_rows(handle["Pos"], [1, 2, 3])
        fancy = np.asarray(handle["Pos"][[1, 2, 3]])

    assert np.array_equal(sliced, fancy)


def test_multimodal_nf_derived_cache_matches_hdf5_samples_and_metadata(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    raw_dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_path=str(codebook_path),
        split_mode="frame_debug",
        split="train",
        split_ratios=(1.0, 0.0, 0.0),
        enabled_modalities=["image", "lidar"],
        seq_len=3,
        num_pred=2,
        return_metadata=True,
    )
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image", "lidar"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    runtime_before_read = None
    cached_dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_path=str(codebook_path),
        split_mode="frame_debug",
        split="train",
        split_ratios=(1.0, 0.0, 0.0),
        enabled_modalities=["image", "lidar"],
        seq_len=3,
        num_pred=2,
        return_metadata=True,
        derived_cache={"image": {"policy": "read_only"}, "lidar": {"policy": "read_only"}},
    )
    runtime_before_read = cached_dataset.derived_cache_runtime_metadata()

    raw = raw_dataset[0]
    cached = cached_dataset[0]
    assert torch.equal(raw["image"], cached["image"])
    assert torch.equal(raw["lidar"], cached["lidar"])
    assert torch.equal(raw["target_beam"], cached["target_beam"])
    assert raw["metadata"]["history_frame_ids"] == cached["metadata"]["history_frame_ids"]
    assert set(prewarm["modalities"]) == {"image", "lidar"}
    for modality in ("image", "lidar"):
        cache_info = cached_dataset.derived_cache_metadata[modality]
        assert cache_info["enabled"] is True
        assert cache_info["source_kind"] == "derived_cache"
        source = next(iter(cache_info["sources"].values()))
        assert source["cache_hit"] is True
        assert cache_info["cache_path_count"] == 1
        assert cache_info["cache_total_bytes"] > 0
        assert cache_info["storage_kind"] == "npy_mmap"
        assert cache_info["layout"] == "source_contiguous_rows"
        assert cache_info["validation_mode"] == "lightweight"
        assert cache_info["source_fingerprint_scanned"] is False
        assert runtime_before_read[modality]["io"]["opened_files"] == 0
        assert cached_dataset.derived_cache_runtime_metadata()[modality]["io"]["opened_files"] == 1
        metadata = json.loads(sidecar_path(source["cache_path"]).read_text(encoding="utf-8"))
        assert metadata["modality"] == modality
        assert metadata["cache_schema_version"] == 2
        assert metadata["source_key"] == channel_path.stem
        assert metadata["storage_kind"] == "npy_mmap"
        assert metadata["layout"] == "source_contiguous_rows"
        assert metadata["bytes"] > 0
        assert metadata["seq_len"] == 3
        assert metadata["num_pred"] == 2
        assert metadata["sample_count"] == 8
        assert "shape" in metadata
        assert "dtype" in metadata


def test_multimodal_nf_read_only_lightweight_validation_skips_source_fingerprint(monkeypatch, tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )

    def fail_fingerprint(_path):
        raise AssertionError("read_only lightweight validation must not fingerprint source HDF5")

    monkeypatch.setattr(derived_cache_module, "fingerprint_path", fail_fingerprint)

    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_path=str(codebook_path),
        split_mode="frame_debug",
        split="train",
        split_ratios=(1.0, 0.0, 0.0),
        enabled_modalities=["image"],
        seq_len=3,
        num_pred=2,
        derived_cache={"image": {"policy": "read_only", "validation_mode": "lightweight"}},
    )

    source = next(iter(dataset.derived_cache_metadata["image"]["sources"].values()))
    assert source["source_fingerprint_scanned"] is False
    assert source["validation_mode"] == "lightweight"


def test_multimodal_nf_strong_validation_detects_fingerprint_mismatch(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    source = prewarm["modalities"]["image"]["sources"][0]
    metadata_path = sidecar_path(source["cache_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_fingerprint"] = "not-the-current-source"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="source_fingerprint"):
        MultimodalNFDataset(
            data_root=str(tmp_path),
            channel_path=str(channel_path),
            codebook_path=str(codebook_path),
            split_mode="frame_debug",
            split="train",
            split_ratios=(1.0, 0.0, 0.0),
            enabled_modalities=["image"],
            seq_len=3,
            num_pred=2,
            derived_cache={"image": {"policy": "read_only", "validation_mode": "strong"}},
        )


def test_multimodal_nf_old_sidecar_read_only_errors_and_auto_upgrades_metadata_only(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    source = prewarm["modalities"]["image"]["sources"][0]
    cache_path = Path(source["cache_path"])
    metadata_path = sidecar_path(source["cache_path"])
    original_bytes = cache_path.read_bytes()
    original_mtime = cache_path.stat().st_mtime_ns
    _downgrade_sidecar_to_v1(metadata_path)

    common = {
        "data_root": str(tmp_path),
        "channel_path": str(channel_path),
        "codebook_path": str(codebook_path),
        "split_mode": "frame_debug",
        "split": "train",
        "split_ratios": (1.0, 0.0, 0.0),
        "enabled_modalities": ["image"],
        "seq_len": 3,
        "num_pred": 2,
    }
    with pytest.raises(FileNotFoundError, match="sidecar migration is pending|preprocess.py"):
        MultimodalNFDataset(**common, derived_cache={"image": {"policy": "read_only"}})
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["version"] == "multimodal_nf_derived_v1"

    upgraded = MultimodalNFDataset(**common, derived_cache={"image": {"policy": "auto"}})
    upgraded_source = next(iter(upgraded.derived_cache_metadata["image"]["sources"].values()))
    assert upgraded_source["metadata_upgraded"] is True
    assert upgraded_source["cache_generated"] is False
    assert upgraded_source["cache_rebuilt"] is False
    assert upgraded_source["migration_pending"] is True
    assert cache_path.read_bytes() == original_bytes
    assert cache_path.stat().st_mtime_ns == original_mtime
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["cache_schema_version"] == 2


def test_multimodal_nf_old_sidecar_status_and_prewarm_upgrade_summary(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    source = prewarm["modalities"]["image"]["sources"][0]
    cache_path = Path(source["cache_path"])
    metadata_path = sidecar_path(cache_path)
    _downgrade_sidecar_to_v1(metadata_path)
    expected = build_expected_metadata(
        source_path=channel_path,
        modality="image",
        profile="rgb_imagenet",
        split="train",
        seq_len=3,
        num_pred=2,
    )

    status = cache_status(cache_path=cache_path, expected=expected, validation_mode="lightweight")
    assert status["status"] == "migration_pending"
    assert status["migration_pending"] is True
    assert status["sidecar_schema_version"] == 1
    assert "cache_schema_version" in status["pending_fields"]

    upgraded = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=False,
        validation_mode="lightweight",
    )
    image_summary = upgraded["modalities"]["image"]
    assert image_summary["metadata_upgraded"] == 1
    assert image_summary["generated"] == 0
    assert image_summary["rebuilt"] == 0
    assert image_summary["failed"] == 0
    assert image_summary["valid"] == 1


def test_multimodal_nf_old_sidecar_strong_validation_rejects_fingerprint_mismatch(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    source = prewarm["modalities"]["image"]["sources"][0]
    metadata_path = sidecar_path(source["cache_path"])
    metadata = _downgrade_sidecar_to_v1(metadata_path)
    metadata["source_fingerprint"] = "not-the-current-source"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    expected = build_expected_metadata(
        source_path=channel_path,
        modality="image",
        profile="rgb_imagenet",
        split="train",
        seq_len=3,
        num_pred=2,
    )

    status = cache_status(cache_path=source["cache_path"], expected=expected, validation_mode="strong")
    assert status["status"] == "invalid"
    assert status["validation"]["source_fingerprint_scanned"] is True
    assert "source_fingerprint" in status["migration_mismatches"]


def test_multimodal_nf_old_sidecar_shape_mismatch_rejects_metadata_upgrade(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    prewarm = prewarm_multimodal_nf_derived_cache(
        data_root=tmp_path,
        channel_path=channel_path,
        modalities=["image"],
        split="train",
        seq_len=3,
        num_pred=2,
        rebuild=True,
    )
    source = prewarm["modalities"]["image"]["sources"][0]
    metadata_path = sidecar_path(source["cache_path"])
    metadata = _downgrade_sidecar_to_v1(metadata_path)
    metadata["shape"] = [999, 5, 6, 3]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    common = {
        "data_root": str(tmp_path),
        "channel_path": str(channel_path),
        "codebook_path": str(codebook_path),
        "split_mode": "frame_debug",
        "split": "train",
        "split_ratios": (1.0, 0.0, 0.0),
        "enabled_modalities": ["image"],
        "seq_len": 3,
        "num_pred": 2,
    }
    with pytest.raises(FileNotFoundError, match="metadata_mismatch|shape"):
        MultimodalNFDataset(**common, derived_cache={"image": {"policy": "read_only"}})

    auto_dataset = MultimodalNFDataset(**common, derived_cache={"image": {"policy": "auto"}})
    auto_source = next(iter(auto_dataset.derived_cache_metadata["image"]["sources"].values()))
    assert auto_source["metadata_upgraded"] is False
    assert auto_source["cache_generated"] is True
    assert auto_source["cache_rebuilt"] is True
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["shape"] == [8, 5, 6, 3]


def test_multimodal_nf_derived_cache_policies_missing_mismatch_auto_and_rebuild(tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)
    common = {
        "data_root": str(tmp_path),
        "channel_path": str(channel_path),
        "codebook_path": str(codebook_path),
        "split_mode": "frame_debug",
        "split": "train",
        "split_ratios": (1.0, 0.0, 0.0),
        "enabled_modalities": ["image"],
        "seq_len": 3,
        "num_pred": 2,
    }

    with pytest.raises(FileNotFoundError, match="image.*read_only"):
        MultimodalNFDataset(**common, derived_cache={"image": {"policy": "read_only"}})

    auto_dataset = MultimodalNFDataset(**common, derived_cache={"image": {"policy": "auto"}})
    auto_source = next(iter(auto_dataset.derived_cache_metadata["image"]["sources"].values()))
    assert auto_source["cache_generated"] is True
    cache_path = Path(auto_source["cache_path"])

    metadata_path = sidecar_path(cache_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seq_len"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metadata_mismatch"):
        MultimodalNFDataset(**common, derived_cache={"image": {"policy": "read_only"}})

    rebuild_dataset = MultimodalNFDataset(**common, derived_cache={"image": {"policy": "rebuild"}})
    rebuild_source = next(iter(rebuild_dataset.derived_cache_metadata["image"]["sources"].values()))
    assert rebuild_source["cache_generated"] is True
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["seq_len"] == 3


def test_multimodal_nf_disabled_modalities_ignore_derived_cache_config_and_missing_large_files(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        image_path=str(tmp_path / "missing_image.h5"),
        lidar_path=str(tmp_path / "missing_lidar.h5"),
        codebook_shape=list(CODEBOOK_SHAPE),
        split_mode="city",
        split="train",
        enabled_modalities=["gps", "csi"],
        seq_len=3,
        num_pred=2,
        derived_cache={
            "image": {"policy": "read_only", "path": str(tmp_path / "missing_image.npy")},
            "lidar": {"policy": "read_only", "path": str(tmp_path / "missing_lidar.npy")},
        },
    )

    sample = dataset[0]
    assert set(sample) >= {"gps", "csi", "target_beam"}
    assert dataset.derived_cache_metadata == {}


def test_multimodal_nf_inactive_modalities_do_not_resolve_derived_cache(monkeypatch, tmp_path: Path):
    channel_path, codebook_path = _write_fixture(tmp_path, sequence=True)

    def fail_cache_status(**_kwargs):
        raise AssertionError("inactive image/lidar modalities must not resolve derived cache")

    monkeypatch.setattr(multimodal_nf_module, "cache_status", fail_cache_status)
    dataset = MultimodalNFDataset(
        data_root=str(tmp_path),
        channel_path=str(channel_path),
        codebook_path=str(codebook_path),
        split_mode="frame_debug",
        split="train",
        split_ratios=(1.0, 0.0, 0.0),
        enabled_modalities=["gps"],
        seq_len=3,
        num_pred=2,
        derived_cache={"image": {"policy": "read_only"}, "lidar": {"policy": "read_only"}},
    )

    assert dataset.derived_cache_metadata == {}


def test_multimodal_nf_index_and_data_factory_skip_csv_requirements(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    outputs = build_multimodal_nf_index(
        data_root=tmp_path,
        channel_path=channel_path,
        cache_dir=tmp_path / "cache",
        split_mode="frame_debug",
        seq_len=3,
        num_pred=2,
        split_ratios=(1.0, 0.0, 0.0),
    )
    assert Path(outputs["index_train"]).exists()
    assert Path(outputs["split_metadata"]).exists()

    cfg = _cfg(tmp_path, channel_path, task="csi", modalities=["csi"])
    dataset = build_dataset(cfg, "train")
    assert isinstance(dataset, MultimodalNFDataset)
    assert dataset.input_profiles == {"csi": "xl_mimo_nf"}
    assert len(dataset) > 0


def test_multimodal_nf_near_field_runtime_metadata_uses_codebook_schema(tmp_path: Path):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    cfg = _cfg(tmp_path, channel_path, task="csi", modalities=["csi"])
    dataset = build_dataset(cfg, "train")
    setup = prediction_setup_metadata(cfg, split_metadata={"train": dataset_run_metadata(dataset)})

    assert setup["objective"] == "near_field_beam_selection"
    assert setup["variant"] == "multimodal_nf_current_frame"
    assert setup["task_semantics"] == "current_frame_near_field_codebook_beam_selection"
    assert setup["legacy_task_semantics"] == "future_near_field_beam_prediction"
    assert setup["target_schema"] == "near_field_3d_codebook_flattened_beam_class"
    assert "near_field_beam_selection" in setup["target_schema_aliases"]
    assert setup["codebook_shape"] == list(CODEBOOK_SHAPE)
    assert setup["flatten_order"] == "azimuth_elevation_range"
    assert setup["num_beam_classes"] == 24
    assert setup["target_fields"] == {"beam_selection": "target_beam"}
    assert setup["output_fields"] == {"beam_selection": "logits"}
    assert setup["dataset_family"]["dataset_type"] == "multimodal_nf"
    assert setup["dataset_family"]["input_profiles"] == {"csi": "xl_mimo_nf"}


@pytest.mark.parametrize(
    ("objective", "aux_heads", "expected_schema", "expected_targets", "expected_outputs"),
    [
        (
            "current_los_classification",
            {"enabled": True, "los": True},
            "los_binary_classification",
            {"los": "los_label"},
            {"los": "los_logits"},
        ),
        (
            "current_link_quality",
            {"enabled": True, "link_quality": True},
            "link_quality_regression",
            {"link_quality": "link_quality"},
            {"link_quality": "link_quality"},
        ),
        (
            "selection_multitask",
            {"enabled": True, "los": True, "link_quality": True},
            "selection_multitask_current_frame",
            {"beam_selection": "target_beam", "los": "los_label", "link_quality": "link_quality"},
            {"beam_selection": "logits", "los": "los_logits", "link_quality": "link_quality"},
        ),
    ],
)
def test_multimodal_nf_auxiliary_objective_runtime_metadata(
    tmp_path: Path,
    objective: str,
    aux_heads: dict[str, bool],
    expected_schema: str,
    expected_targets: dict[str, str],
    expected_outputs: dict[str, str],
):
    channel_path, _ = _write_fixture(tmp_path, sequence=True)
    cfg = _cfg(tmp_path, channel_path, task="csi", modalities=["csi"])
    cfg["experiment"]["objective"] = objective
    cfg["model"]["student"]["auxiliary_heads"] = aux_heads

    setup = prediction_setup_metadata(cfg)

    assert setup["objective"] == objective
    assert setup["target_schema"] == expected_schema
    assert setup["task_semantics"] != "future_near_field_beam_prediction"
    assert setup["target_fields"] == expected_targets
    assert setup["output_fields"] == expected_outputs
    assert set(setup["loss_fields"])
    assert set(setup["metric_fields"])
    if objective == "selection_multitask":
        assert setup["targets"]["beam_selection"]["schema"] == "near_field_3d_codebook_flattened_beam_class"
        assert setup["targets"]["los"]["schema"] == "los_binary_classification"
        assert setup["targets"]["link_quality"]["schema"] == "link_quality_regression"


def test_multimodal_nf_index_uses_all_city_files_and_pairs_modalities(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    _write_fixture_h5(raw / "City_A_dataset.h5", city_values=[b"City_A"] * 6, trajectory_values=[0] * 6)
    _write_fixture_h5(raw / "City_B_dataset.h5", city_values=[b"City_B"] * 6, trajectory_values=[0] * 6)
    _write_fixture_h5(raw / "City_A_img.h5", city_values=[b"City_A"] * 6, trajectory_values=[0] * 6)
    _write_fixture_h5(raw / "City_B_lidar.h5", city_values=[b"City_B"] * 6, trajectory_values=[0] * 6)

    outputs = build_multimodal_nf_index(data_root=tmp_path, cache_dir=tmp_path / "cache", split_mode="city", seq_len=3, num_pred=2)
    index = SampleIndex.from_csv(outputs["index_all"], storage_kind="hdf5_frame")

    assert len(index) == 4
    assert {row.scene_or_city for row in index.rows} == {"City_A", "City_B"}
    assert any(row.resource_refs.get("image_path", "").endswith("City_A_img.h5") for row in index.rows)
    assert any(row.resource_refs.get("lidar_path", "").endswith("City_B_lidar.h5") for row in index.rows)
    assert all(len(row.resource_refs["history_indices"]) == 3 for row in index.rows)
    assert all(len(row.target_ref["target_indices"]) == 2 for row in index.rows)


def test_runtime_dataset_flat_contract_and_profile_batch_validation():
    class DummyAdapter:
        modality = "gps"
        profile = "uav_xyz_snapshot"
        sample_key = "gps"

        def load(self, row):
            return {"gps": torch.ones(1, 3)}

        def metadata(self):
            return {"modality": self.modality, "profile": self.profile}

    class DummyTarget:
        target_schema = "near_field_beam_selection"

        def load(self, row):
            return {"target_beam": torch.tensor([1])}

        def metadata(self):
            return {"target_schema": self.target_schema}

    index = SampleIndex.from_rows(
        [SampleRow(sample_id="s0", split="train", dataset_type="dummy", family="Dummy")],
        storage_kind="hdf5_frame",
        metadata={"split": "train"},
    )
    dataset = RuntimeDataset(
        sample_index=index,
        modality_adapters=[DummyAdapter()],
        target_provider=DummyTarget(),
        dataset_type="dummy",
        descriptor={"family": "Dummy", "storage_kind": "hdf5_frame"},
        enabled_modalities=["gps"],
        input_profiles={"gps": "uav_xyz_snapshot"},
        return_metadata=True,
    )
    sample = dataset[0]
    assert sample["gps"].shape == (1, 3)
    assert sample["target_beam"].shape == (1,)
    assert sample["metadata"]["sample_id"] == "s0"

    lidar_batch = {"lidar": torch.zeros(2, 1, 10, 3)}
    assert prepare_lidar_inputs(
        lidar_batch,
        seq_length=1,
        num_pred=1,
        device=torch.device("cpu"),
        profile="point_cloud_xyz_10000",
    ).shape == (2, 1, 10, 3)
    csi_batch = {"csi": torch.zeros(2, 1, 4, 2, 2)}
    assert prepare_csi_inputs(
        csi_batch,
        seq_length=1,
        num_pred=1,
        device=torch.device("cpu"),
        profile="xl_mimo_nf",
    ).shape == (2, 1, 4, 2, 2)


def test_multimodal_nf_profile_resolution_rejects_unknown():
    assert resolve_dataset_profiles(
        "multimodal_nf",
        ["gps", "csi"],
        {},
    ) == {"gps": "uav_xyz_snapshot", "csi": "xl_mimo_nf"}
    with pytest.raises(ValueError, match="unknown_channel"):
        resolve_dataset_profiles("multimodal_nf", ["csi"], {"csi_profile": "unknown_channel"})


def _cfg(tmp_path: Path, channel_path: Path, *, task: str, modalities: list[str]) -> dict:
    return {
        "experiment": {"task": task, "objective": "near_field_beam_selection"},
        "data": {
            "dataset": {
                "type": "multimodal_nf",
                "data_root": str(tmp_path),
                "channel_path": str(channel_path),
                "codebook_shape": list(CODEBOOK_SHAPE),
                "split_mode": "frame_debug",
                "seq_len": 3,
                "num_pred": 2,
            },
            "dataloader": {"batch_size": 2, "num_workers": 0},
        },
        "model": {
            "modalities": modalities,
            "num_classes": 24,
            "num_pred": 2,
            "seq_length_student": 3,
            "student": {
                "type": "modular_sequence",
                "modalities": modalities,
                "num_classes": 24,
                "num_pred": 2,
                "input_profiles": {name: dataset_descriptor("multimodal_nf").profile_for(name).default_profile for name in modalities},
            },
        },
    }


def _downgrade_sidecar_to_v1(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "cache_schema_version",
        "source_key",
        "source_size_bytes",
        "source_mtime_ns",
        "storage_kind",
        "layout",
        "bytes",
        "recommended_access_pattern",
        "metadata_upgraded_at",
        "previous_cache_schema_version",
    ):
        metadata.pop(key, None)
    metadata["version"] = "multimodal_nf_derived_v1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _write_fixture(tmp_path: Path, *, bad_lidar: bool = False, sequence: bool = False) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir(parents=True, exist_ok=True)
    channel_path = raw / "City_fixture.h5"
    if sequence:
        _write_fixture_h5(
            channel_path,
            bad_lidar=bad_lidar,
            n=8,
            city_values=[b"City_A"] * 8,
            trajectory_values=[0] * 8,
            frame_values=list(range(8)),
        )
    else:
        _write_fixture_h5(channel_path, bad_lidar=bad_lidar)
    codebook_path = codebooks / "small_codebook.pkl"
    with codebook_path.open("wb") as handle:
        pickle.dump({"shape": CODEBOOK_SHAPE, "name": "fixture_small"}, handle)
    return channel_path, codebook_path


def _write_fixture_h5(
    path: Path,
    *,
    include_beam_power: bool = True,
    bad_lidar: bool = False,
    city_values=None,
    trajectory_values=None,
    frame_values=None,
    n: int = 6,
    include_traj_nlos: bool = True,
    include_mode: bool = True,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("H", data=np.arange(n * 4 * 2 * 2, dtype=np.float32).reshape(n, 4, 2, 2))
        handle.create_dataset("Pos", data=np.arange(n * 3, dtype=np.float32).reshape(n, 3))
        topk = np.zeros((n, 5, 3), dtype=np.int64)
        for idx in range(n):
            for rank in range(5):
                topk[idx, rank] = [(idx + rank) % 2, (idx + rank) % 3, (idx + rank) % 4]
        handle.create_dataset("BeamIdx", data=topk)
        if include_beam_power:
            handle.create_dataset("BeamPower", data=np.linspace(1.0, 0.1, n * 5, dtype=np.float32).reshape(n, 5))
        handle.create_dataset("Has_LoS", data=np.asarray(([1, 0, 1, 1, 0, 1, 1, 0] * ((n + 7) // 8))[:n], dtype=np.int64))
        handle.create_dataset("Is_NF", data=np.asarray(([1, 1, 0, 1, 0, 1, 1, 1] * ((n + 7) // 8))[:n], dtype=np.int64))
        handle.create_dataset(
            "City",
            data=np.asarray(city_values or [b"City_A", b"City_A", b"City_B", b"City_B", b"City_C", b"City_C"][:n]),
        )
        handle.create_dataset("Trajectory", data=np.asarray(trajectory_values or [0, 0, 1, 1, 2, 2][:n], dtype=np.int64))
        handle.create_dataset("Frame", data=np.asarray(frame_values or [0, 1, 0, 1, 0, 1][:n], dtype=np.int64))
        if include_traj_nlos:
            handle.create_dataset(
                "Traj_Is_NLoS",
                data=np.asarray(([0, 1, 0, 0, 1, 0, 1, 0] * ((n + 7) // 8))[:n], dtype=np.int64),
            )
        if include_mode:
            handle.create_dataset("Mode_Idx", data=np.asarray(([0, 1, 0, 1, 0, 1, 0, 1] * ((n + 7) // 8))[:n], dtype=np.int64))
        handle.create_dataset("image", data=np.arange(n * 5 * 6 * 3, dtype=np.uint8).reshape(n, 5, 6, 3))
        lidar_shape = (n, 10, 2) if bad_lidar else (n, 10, 3)
        handle.create_dataset("lidar", data=np.arange(np.prod(lidar_shape), dtype=np.float32).reshape(lidar_shape))
