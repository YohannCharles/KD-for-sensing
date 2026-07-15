
import datetime as dt
import json
from pathlib import Path
import shutil
import stat
from typing import Any

from tqdm.auto import tqdm

from kd_sensing.data.sample_cache import LmdbSampleCache, sample_cache_path_for_split
from kd_sensing.engine.data_factory_scalers import normalization_kwargs
from kd_sensing.registries import DATASETS, PREPROCESSORS
from kd_sensing.utils.paths import project_root, resolve_path


SAMPLE_CACHE_MARKER = ".kd_sensing_sample_cache.json"
SAMPLE_CACHE_MARKER_SCHEMA_VERSION = 1
_SAMPLE_CACHE_FILES = {SAMPLE_CACHE_MARKER, "data.mdb", "lock.mdb"}


def generate_sample_lmdb_cache(
    *,
    dataset: dict[str, Any],
    path: str,
    splits: list[str] | tuple[str, ...] = ("train", "test"),
    map_size_gb: float = 64.0,
    overwrite: bool = False,
    progress: bool = True,
    cache_type: str = "sample_lmdb_cache",
    cache_root: str | None = None,
) -> dict[str, Any]:
    reports = []
    train_dataset = None
    for split in splits:
        split_name = str(split)
        raw_cache_path = _raw_cache_path(path, split_name)
        allowed_root = resolve_path(cache_root) if cache_root is not None else project_root().resolve()
        cache_path = sample_cache_path_for_split(path, split_name)
        _validate_cache_target(raw_cache_path, cache_path, allowed_root)
        if cache_path.exists() and overwrite:
            _validate_owned_cache(cache_path, allowed_root=allowed_root, owner=cache_type)
            shutil.rmtree(cache_path)
        ds_cfg = dict(dataset)
        ds_cfg["split"] = split_name
        ds_cfg["sample_cache"] = None
        if split_name != "train" and train_dataset is not None:
            ds_cfg.update(normalization_kwargs(train_dataset))
        ds = DATASETS.build(ds_cfg)
        if split_name == "train":
            train_dataset = ds
        cache_path.mkdir(parents=True, exist_ok=True)
        _write_cache_marker(cache_path, owner=cache_type)
        cache = LmdbSampleCache(cache_path, readonly=False, map_size_gb=map_size_gb)
        iterator = range(len(ds))
        if progress:
            iterator = tqdm(iterator, desc=f"Sample LMDB {split_name}", unit="sample")
        written = 0
        try:
            for idx in iterator:
                key_builder = getattr(ds, "_sample_cache_key", None)
                key = key_builder(int(idx)) if callable(key_builder) else f"{split_name}:{idx}"
                cache.put(key, ds[int(idx)])
                written += 1
            cache.put_metadata(
                {
                    "type": cache_type,
                    "dataset_type": str(dataset.get("type", "deepsense6g")),
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "split": split_name,
                    "count": int(written),
                    "root_csv": str(ds.root_csv),
                    "data_root": str(ds.data_root),
                    "enabled_modalities": list(ds.enabled_modalities),
                    "seq_len": int(ds.seq_len),
                    "num_pred": int(ds.num_pred),
                    "gps_feature_mode": getattr(ds, "gps_feature_mode", None),
                    "gps_angle_frame": getattr(ds, "gps_angle_frame", None),
                    "gps_yaw_source": getattr(ds, "gps_yaw_source", None),
                    "condition": str(getattr(ds, "condition", "")),
                    "scenario": str(getattr(ds, "scene_slug", getattr(ds, "scene_id", ""))),
                }
            )
        finally:
            cache.close()
        reports.append({"split": split_name, "path": str(cache_path), "count": int(written)})
    return {"type": cache_type, "reports": reports}


def _raw_cache_path(path: str, split: str) -> Path:
    candidate = Path(str(path).format(split=str(split))).expanduser()
    return candidate if candidate.is_absolute() else project_root() / candidate


def _validate_cache_target(raw_path: Path, cache_path: Path, allowed_root: Path) -> None:
    root = project_root().resolve()
    target = cache_path.resolve()
    allowed = allowed_root.resolve()
    symlink = _first_symlink_component(raw_path)
    if symlink is not None:
        raise ValueError(f"Sample cache path must not contain a symlink: {symlink}")
    protected = {root, root / "dataset", root / "outputs"}
    if target in protected or target == allowed or not target.is_relative_to(allowed):
        raise ValueError(f"Unsafe sample cache path {target}; expected a child of {allowed}.")


def _first_symlink_component(path: Path) -> Path | None:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = project_root() / absolute
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def _validate_owned_cache(cache_path: Path, *, allowed_root: Path, owner: str) -> None:
    _validate_cache_target(cache_path, cache_path, allowed_root)
    marker_path = cache_path / SAMPLE_CACHE_MARKER
    if not marker_path.is_file():
        raise ValueError(f"Refusing to overwrite unowned sample cache without {SAMPLE_CACHE_MARKER}: {cache_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid sample cache ownership marker: {marker_path}") from exc
    expected = {
        "schema_version": SAMPLE_CACHE_MARKER_SCHEMA_VERSION,
        "owner": owner,
        "path": str(cache_path.resolve()),
    }
    mismatches = [key for key, value in expected.items() if marker.get(key) != value]
    unexpected = sorted(item.name for item in cache_path.iterdir() if item.name not in _SAMPLE_CACHE_FILES)
    structure_errors = [
        name
        for name in ("data.mdb", "lock.mdb")
        if not _is_regular_non_symlink_file(cache_path / name)
    ]
    if mismatches or unexpected or structure_errors:
        detail = ", ".join(mismatches)
        if unexpected:
            detail = detail or f"unexpected entries: {unexpected}"
        if structure_errors:
            detail = detail or f"invalid LMDB structure: {structure_errors} must be regular non-symlink files"
        raise ValueError(f"Refusing to overwrite sample cache with mismatched ownership: {detail}")


def _is_regular_non_symlink_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) and not stat.S_ISLNK(mode)


def _write_cache_marker(cache_path: Path, *, owner: str) -> None:
    marker_path = cache_path / SAMPLE_CACHE_MARKER
    payload = {
        "schema_version": SAMPLE_CACHE_MARKER_SCHEMA_VERSION,
        "owner": owner,
        "path": str(cache_path.resolve()),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker_path)


def generate_deepsense6g_sample_lmdb_cache(**kwargs: Any) -> dict[str, Any]:
    return generate_sample_lmdb_cache(cache_type="deepsense6g_sample_lmdb_cache", **kwargs)


@PREPROCESSORS.register("sample_lmdb_cache")
class SampleLMDBCachePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return generate_sample_lmdb_cache(**self.kwargs)


@PREPROCESSORS.register("deepsense6g_sample_lmdb_cache")
class DeepSense6GSampleLMDBCachePreprocessor:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def run(self):
        return generate_deepsense6g_sample_lmdb_cache(**self.kwargs)
