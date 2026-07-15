from pathlib import Path

import pandas as pd
import pytest
import torch

import kd_sensing.preprocessing.image_cache as image_cache_module
from kd_sensing.preprocessing.image_cache import prewarm_image_derived_cache


def _write_image_csv(root: Path, paths: list[str]) -> Path:
    csv_path = root / "sequences.csv"
    pd.DataFrame({"camera1": paths}).to_csv(csv_path, index=False)
    return csv_path


def _stub_image_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_cache_module, "read_image_array", lambda path: path.read_bytes())
    monkeypatch.setattr(
        image_cache_module,
        "build_rgb_imagenet_transform",
        lambda image_size: lambda image: torch.zeros((3, int(image_size[0]), int(image_size[1]))),
    )


def test_image_cache_prewarm_rejects_data_root_overlap(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    csv_path = _write_image_csv(data_root, ["camera/one.jpg"])
    sentinel = data_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="must be disjoint"):
        prewarm_image_derived_cache(
            csv_path=csv_path,
            data_root=data_root,
            cache_dir=data_root,
            progress=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_image_cache_prewarm_failure_preserves_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _stub_image_transform(monkeypatch)
    data_root = tmp_path / "data"
    camera_dir = data_root / "camera"
    camera_dir.mkdir(parents=True)
    (camera_dir / "valid.jpg").write_bytes(b"valid")
    csv_path = _write_image_csv(data_root, ["camera/valid.jpg", "camera/missing.jpg"])
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "sentinel.txt").write_text("old-cache", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exceeded max_failure_rate"):
        prewarm_image_derived_cache(
            csv_path=csv_path,
            data_root=data_root,
            cache_dir=cache_root,
            progress=False,
        )

    assert (cache_root / "sentinel.txt").read_text(encoding="utf-8") == "old-cache"
    assert sorted(path.name for path in cache_root.iterdir()) == ["sentinel.txt"]


def test_image_cache_prewarm_publishes_only_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _stub_image_transform(monkeypatch)
    data_root = tmp_path / "data"
    camera_dir = data_root / "camera"
    camera_dir.mkdir(parents=True)
    (camera_dir / "valid.jpg").write_bytes(b"valid")
    csv_path = _write_image_csv(data_root, ["camera/valid.jpg"])
    cache_root = tmp_path / "cache"

    report = prewarm_image_derived_cache(
        csv_path=csv_path,
        data_root=data_root,
        cache_dir=cache_root,
        progress=False,
    )

    assert report["generated"] == 1
    assert (cache_root / "prewarm_report.json").exists()
    assert len(list(cache_root.rglob("*.npy"))) == 1
