from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from kd_sensing.utils.paths import resolve_path


class SampleCacheDependencyError(RuntimeError):
    pass


def _lmdb():
    try:
        import lmdb
    except ImportError as exc:
        raise SampleCacheDependencyError(
            "LMDB sample cache requires the 'lmdb' package. "
            "Install it with: conda run -n kd_mm_beam python -m pip install lmdb"
        ) from exc
    return lmdb


class LmdbSampleCache:
    def __init__(
        self,
        path: str | Path,
        *,
        readonly: bool = True,
        map_size_gb: float = 64.0,
        lock: bool | None = None,
        readahead: bool = True,
    ) -> None:
        self.path = resolve_path(path)
        self.readonly = bool(readonly)
        self.map_size = int(float(map_size_gb) * 1024**3)
        self.lock = not self.readonly if lock is None else bool(lock)
        self.readahead = bool(readahead)
        self._env = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path
        if self.readonly and not path.exists():
            return None
        with self._open().begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
        return None if raw is None else pickle.loads(raw)

    def put(self, key: str, sample: dict[str, Any]) -> None:
        if self.readonly:
            raise RuntimeError("Cannot write to a read-only LMDB sample cache.")
        with self._open().begin(write=True) as txn:
            txn.put(key.encode("utf-8"), pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))

    def put_metadata(self, metadata: dict[str, Any]) -> None:
        if self.readonly:
            raise RuntimeError("Cannot write metadata to a read-only LMDB sample cache.")
        with self._open().begin(write=True) as txn:
            txn.put(b"__metadata__", pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))

    def _open(self):
        if self._env is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._env = _lmdb().open(
                str(self.path),
                subdir=True,
                readonly=self.readonly,
                create=not self.readonly,
                lock=self.lock,
                readahead=self.readahead,
                map_size=self.map_size,
                max_readers=2048,
            )
        return self._env


def sample_cache_path_for_split(path: str | Path, split: str) -> Path:
    return resolve_path(str(path).format(split=str(split)))


__all__ = ["LmdbSampleCache", "SampleCacheDependencyError", "sample_cache_path_for_split"]
