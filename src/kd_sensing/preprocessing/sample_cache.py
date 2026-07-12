
import datetime as dt
from typing import Any

from tqdm.auto import tqdm

from kd_sensing.data.sample_cache import LmdbSampleCache, sample_cache_path_for_split
from kd_sensing.engine.data_factory_scalers import normalization_kwargs
from kd_sensing.registries import DATASETS, PREPROCESSORS


def generate_sample_lmdb_cache(
    *,
    dataset: dict[str, Any],
    path: str,
    splits: list[str] | tuple[str, ...] = ("train", "test"),
    map_size_gb: float = 64.0,
    overwrite: bool = False,
    progress: bool = True,
    cache_type: str = "sample_lmdb_cache",
) -> dict[str, Any]:
    reports = []
    train_dataset = None
    for split in splits:
        split_name = str(split)
        cache_path = sample_cache_path_for_split(path, split_name)
        if cache_path.exists() and overwrite:
            import shutil

            shutil.rmtree(cache_path)
        ds_cfg = dict(dataset)
        ds_cfg["split"] = split_name
        ds_cfg["sample_cache"] = None
        if split_name != "train" and train_dataset is not None:
            ds_cfg.update(normalization_kwargs(train_dataset))
        ds = DATASETS.build(ds_cfg)
        if split_name == "train":
            train_dataset = ds
        cache = LmdbSampleCache(cache_path, readonly=False, map_size_gb=map_size_gb)
        iterator = range(len(ds))
        if progress:
            iterator = tqdm(iterator, desc=f"Sample LMDB {split_name}", unit="sample")
        written = 0
        try:
            for idx in iterator:
                cache.put(f"{split_name}:{idx}", ds[int(idx)])
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
                    "condition": str(getattr(ds, "condition", "")),
                    "scenario": str(getattr(ds, "scene_slug", getattr(ds, "scene_id", ""))),
                }
            )
        finally:
            cache.close()
        reports.append({"split": split_name, "path": str(cache_path), "count": int(written)})
    return {"type": cache_type, "reports": reports}


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
