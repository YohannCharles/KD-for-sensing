import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from kd_sensing.diagnostics.jepa_benchmark_common import BenchmarkManifestError, _json_ready, _sha256_text


PERTURBATION_CACHE_SCHEMA_VERSION = "benchmark_perturbation_batch_v1"


def real_forward_requested(manifest: Mapping[str, Any], model_spec: Mapping[str, Any]) -> bool:
    model_cfg = model_spec.get("real_forward")
    if isinstance(model_cfg, Mapping) and "enabled" in model_cfg:
        return bool(model_cfg.get("enabled"))
    evaluation = manifest.get("evaluation", {}) if isinstance(manifest.get("evaluation"), Mapping) else {}
    return str(evaluation.get("mode", "")).strip().lower().replace("-", "_") == "real_forward"


def real_forward_settings(manifest: Mapping[str, Any], model_spec: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = manifest.get("evaluation", {}) if isinstance(manifest.get("evaluation"), Mapping) else {}
    base = evaluation.get("real_forward", {}) if isinstance(evaluation.get("real_forward"), Mapping) else {}
    model_rf = model_spec.get("real_forward", {}) if isinstance(model_spec.get("real_forward"), Mapping) else {}
    settings = {**dict(base), **dict(model_rf)}
    settings.setdefault("enabled", True)
    settings.setdefault("resume", True)
    settings.setdefault("cache_subdir", "real_forward")
    if "sample_count" not in settings and "sample_count" in model_spec:
        settings["sample_count"] = model_spec["sample_count"]
    return settings


def perturbation_cache_config(settings: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    raw = settings.get("perturbation_cache", settings.get("perturbed_data_cache", {}))
    if not raw:
        return {"mode": "off", "dir": str(output_dir / "cache" / "perturbations")}
    if isinstance(raw, str):
        raw = {"mode": raw}
    if raw is True:
        raw = {"mode": "read_write"}
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError("evaluation.real_forward.perturbation_cache must be a mapping, string, or boolean.")
    mode = str(raw.get("mode", "read_write")).strip().lower().replace("-", "_")
    if mode not in {"off", "write", "read", "read_write"}:
        raise BenchmarkManifestError("perturbation_cache.mode must be one of off, write, read, or read_write.")
    cache_dir = Path(str(raw.get("dir", raw.get("cache_dir", output_dir / "cache" / "perturbations"))))
    if not cache_dir.is_absolute():
        cache_dir = output_dir / cache_dir
    return {"mode": mode, "dir": str(cache_dir)}


def perturbation_cache_key(condition: Mapping[str, Any], *, split: str, sample_limit: int) -> str:
    payload = {
        "schema": PERTURBATION_CACHE_SCHEMA_VERSION,
        "split": split,
        "sample_limit": int(sample_limit),
        "condition": _json_ready(condition),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))[:24]


def perturbation_cache_index_path(cache_cfg: Mapping[str, Any], key: str) -> Path:
    return Path(str(cache_cfg["dir"])) / key / "index.json"


def load_perturbation_cache_index(path: Path, *, key: str, cache_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        if str(cache_cfg.get("mode")) == "read":
            raise BenchmarkManifestError(f"Perturbation cache missing for key {key}: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PERTURBATION_CACHE_SCHEMA_VERSION or payload.get("key") != key:
        raise BenchmarkManifestError(f"Perturbation cache mismatch for key {key}: {path}")
    return payload


def iter_perturbation_cache(index: Mapping[str, Any]) -> Iterable[tuple[dict[str, Any], list[str], list[dict[str, Any]]]]:
    root = Path(str(index["root"]))
    for shard in index.get("shards", []):
        if not isinstance(shard, Mapping):
            continue
        path = root / str(shard["file"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != PERTURBATION_CACHE_SCHEMA_VERSION:
            raise BenchmarkManifestError(f"Perturbation cache shard schema mismatch: {path}")
        yield dict(payload["batch"]), [str(item) for item in payload.get("sample_ids", [])], list(payload.get("warnings", []))


def write_perturbation_cache_shard(
    index_path: Path,
    *,
    shard_index: int,
    key: str,
    condition: Mapping[str, Any],
    batch: Mapping[str, Any],
    sample_ids: list[str],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    root = index_path.parent
    root.mkdir(parents=True, exist_ok=True)
    name = f"batch_{shard_index:06d}.pt"
    payload = {
        "schema": PERTURBATION_CACHE_SCHEMA_VERSION,
        "key": key,
        "condition": _json_ready(condition),
        "sample_ids": list(sample_ids),
        "warnings": _json_ready(warnings),
        "batch": dict(batch),
    }
    torch.save(payload, root / name)
    return {"file": name, "sample_count": len(sample_ids)}


def write_perturbation_cache_index(
    index_path: Path,
    *,
    key: str,
    condition: Mapping[str, Any],
    split: str,
    sample_limit: int,
    shards: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": PERTURBATION_CACHE_SCHEMA_VERSION,
        "key": key,
        "root": str(index_path.parent),
        "condition": _json_ready(condition),
        "split": split,
        "sample_limit": int(sample_limit),
        "sample_count": int(sum(int(item.get("sample_count", 0)) for item in shards)),
        "shards": shards,
    }
    index_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "PERTURBATION_CACHE_SCHEMA_VERSION",
    "iter_perturbation_cache",
    "load_perturbation_cache_index",
    "perturbation_cache_config",
    "perturbation_cache_index_path",
    "perturbation_cache_key",
    "real_forward_requested",
    "real_forward_settings",
    "write_perturbation_cache_index",
    "write_perturbation_cache_shard",
]
