import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kd_sensing.data.dataset_runtime import SampleRow


SUPPORTED_DOMAIN_TYPES = {"scenario", "weather", "scenario_weather", "town_scenario_weather"}
SUPPORTED_SELECTIONS = {
    "random",
    "stratified_by_beam",
    "stratified_by_geo_sector",
    "stratified_by_weather",
}
ARTIFACT_VERSION = "target_shot_split_v1"


@dataclass(frozen=True)
class TargetShotSplitConfig:
    domain_type: str
    source_domains: tuple[str, ...]
    target_domains: tuple[str, ...]
    target_label_fraction: float = 0.05
    target_label_selection: str = "random"
    seed: int = 42
    allow_target_unlabeled: bool = True
    artifact_path: str | None = None
    target_test_fraction: float = 0.2
    fallback_target_label_selection: str | None = None
    overwrite: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "TargetShotSplitConfig":
        split_cfg = cfg.get("split", cfg) if isinstance(cfg.get("split", cfg), Mapping) else {}
        domain_type = str(split_cfg.get("domain_type", "scenario")).strip().lower()
        if domain_type not in SUPPORTED_DOMAIN_TYPES:
            raise ValueError(f"split.domain_type must be one of {sorted(SUPPORTED_DOMAIN_TYPES)}, got {domain_type!r}.")
        selection = str(split_cfg.get("target_label_selection", "random")).strip().lower()
        if selection not in SUPPORTED_SELECTIONS:
            raise ValueError(
                f"split.target_label_selection must be one of {sorted(SUPPORTED_SELECTIONS)}, got {selection!r}."
            )
        return cls(
            domain_type=domain_type,
            source_domains=tuple(_list_text(split_cfg.get("source_domains"))),
            target_domains=tuple(_list_text(split_cfg.get("target_domains"))),
            target_label_fraction=float(split_cfg.get("target_label_fraction", 0.05)),
            target_label_selection=selection,
            seed=int(split_cfg.get("seed", 42)),
            allow_target_unlabeled=bool(split_cfg.get("allow_target_unlabeled", True)),
            artifact_path=_optional_path_text(split_cfg.get("artifact_path", split_cfg.get("output_path"))),
            target_test_fraction=float(split_cfg.get("target_test_fraction", 0.2)),
            fallback_target_label_selection=_optional_text(split_cfg.get("fallback_target_label_selection")),
            overwrite=bool(split_cfg.get("overwrite", False)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "domain_type": self.domain_type,
            "source_domains": list(self.source_domains),
            "target_domains": list(self.target_domains),
            "target_label_fraction": float(self.target_label_fraction),
            "target_label_selection": self.target_label_selection,
            "seed": int(self.seed),
            "allow_target_unlabeled": bool(self.allow_target_unlabeled),
            "target_test_fraction": float(self.target_test_fraction),
            "fallback_target_label_selection": self.fallback_target_label_selection,
        }


def build_domain_key(row: Mapping[str, Any] | SampleRow, *, domain_type: str, dataset_type: str | None = None) -> str:
    record = _row_record(row)
    domain_type = str(domain_type or "scenario").strip().lower()
    if domain_type not in SUPPORTED_DOMAIN_TYPES:
        raise ValueError(f"Unsupported domain_type={domain_type!r}; expected one of {sorted(SUPPORTED_DOMAIN_TYPES)}.")
    dataset = str(dataset_type or record.get("dataset_type") or record["metadata"].get("dataset_type") or "unknown")
    if domain_type == "scenario":
        return _require(record, "scenario", dataset_type=dataset, domain_type=domain_type)
    if domain_type == "weather":
        return _require_weather(record, dataset_type=dataset, domain_type=domain_type)
    if domain_type == "scenario_weather":
        scenario = _require(record, "scenario", dataset_type=dataset, domain_type=domain_type)
        weather = _require_weather(record, dataset_type=dataset, domain_type=domain_type)
        return f"{scenario}:{weather}"
    town = _require(record, "town", dataset_type=dataset, domain_type=domain_type)
    scenario = _require(record, "scenario", dataset_type=dataset, domain_type=domain_type)
    weather = _require_weather(record, dataset_type=dataset, domain_type=domain_type)
    return f"{town}:{scenario}:{weather}"


def build_target_shot_split(
    rows: Iterable[Mapping[str, Any] | SampleRow],
    config: TargetShotSplitConfig | Mapping[str, Any],
    *,
    dataset_type: str | None = None,
    leakage_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, TargetShotSplitConfig) else TargetShotSplitConfig.from_config(config)
    records = [_row_record(row, index=idx) for idx, row in enumerate(rows)]
    if not records:
        raise ValueError("Cannot build target-shot split from an empty sample list.")
    for record in records:
        record["domain_key"] = build_domain_key(record, domain_type=cfg.domain_type, dataset_type=dataset_type)
    if not cfg.source_domains:
        raise ValueError("split.source_domains must contain at least one domain key.")
    if not cfg.target_domains:
        raise ValueError("split.target_domains must contain at least one domain key.")
    source_set = set(cfg.source_domains)
    target_set = set(cfg.target_domains)
    source = [item for item in records if item["domain_key"] in source_set]
    target = [item for item in records if item["domain_key"] in target_set]
    if not source:
        raise ValueError(f"No source samples matched source_domains={sorted(source_set)}.")
    if not target:
        raise ValueError(f"No target samples matched target_domains={sorted(target_set)}.")
    target_adapt_pool, target_test = _split_target_pool(target, cfg)
    selected, sampling_manifest = _select_target_labeled(target_adapt_pool, cfg)
    labeled_ids = {item["sample_id"] for item in selected}
    target_unlabeled = [item for item in target_adapt_pool if item["sample_id"] not in labeled_ids]
    if not cfg.allow_target_unlabeled:
        target_unlabeled = []
    splits = {
        "source": source,
        "target_labeled": selected,
        "target_unlabeled": target_unlabeled,
        "target_test": target_test,
    }
    diagnostics = _leakage_diagnostics(splits, leakage_metadata=leakage_metadata)
    if diagnostics["sample_id_overlap_count"] > 0:
        raise ValueError(f"Target-shot split sample ids overlap: {diagnostics['sample_id_overlap_examples']}.")
    fingerprint = sample_id_fingerprint(records)
    artifact = {
        "version": ARTIFACT_VERSION,
        "config_summary": cfg.summary(),
        "input_fingerprint": fingerprint,
        "input_sample_count": int(len(records)),
        "dataset_type": str(dataset_type or _first(records, "dataset_type", default="unknown")),
        "domain_metadata": {
            "domain_type": cfg.domain_type,
            "source_domains": list(cfg.source_domains),
            "target_domains": list(cfg.target_domains),
            "domain_counts": dict(sorted(Counter(item["domain_key"] for item in records).items())),
        },
        "splits": {name: _split_payload(items) for name, items in splits.items()},
        "stats": _split_stats(splits),
        "sampling_manifest": sampling_manifest,
        "leakage_diagnostics": diagnostics,
        "strict_eligibility": {
            "eligible": diagnostics["sample_id_overlap_count"] == 0
            and int(diagnostics.get("guard_band_violations", 0) or 0) == 0,
            "reasons": diagnostics.get("eligibility_reasons", []),
        },
        "repair_hint": "Regenerate this artifact or pass split.overwrite=true after intentionally changing inputs.",
    }
    return artifact


def write_target_shot_artifact(artifact: Mapping[str, Any], path: str | Path) -> dict[str, str]:
    json_path = Path(path)
    if json_path.suffix.lower() == ".npz":
        npz_path = json_path
        json_path = json_path.with_suffix(".json")
    else:
        npz_path = json_path.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    payload["npz_path"] = str(npz_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    arrays: dict[str, Any] = {}
    for split_name, split_payload in payload.get("splits", {}).items():
        arrays[f"{split_name}_sample_ids"] = np.asarray(split_payload.get("sample_ids", []), dtype=object)
        arrays[f"{split_name}_indices"] = np.asarray(split_payload.get("indices", []), dtype=np.int64)
    np.savez_compressed(npz_path, **arrays)
    return {"json": str(json_path), "npz": str(npz_path)}


def load_target_shot_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        source = source.with_suffix(".json")
    return json.loads(source.read_text(encoding="utf-8"))


def validate_target_shot_artifact(
    artifact: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any] | SampleRow],
    config: TargetShotSplitConfig | Mapping[str, Any],
) -> None:
    cfg = config if isinstance(config, TargetShotSplitConfig) else TargetShotSplitConfig.from_config(config)
    records = [_row_record(row, index=idx) for idx, row in enumerate(rows)]
    expected = {
        "input_fingerprint": sample_id_fingerprint(records),
        "seed": int(cfg.seed),
        "target_label_fraction": float(cfg.target_label_fraction),
        "source_domains": list(cfg.source_domains),
        "target_domains": list(cfg.target_domains),
    }
    actual_cfg = artifact.get("config_summary", {})
    mismatches = []
    if artifact.get("input_fingerprint") != expected["input_fingerprint"]:
        mismatches.append("input_fingerprint")
    for key in ("seed", "target_label_fraction", "source_domains", "target_domains"):
        if actual_cfg.get(key) != expected[key]:
            mismatches.append(key)
    artifact_ids = _artifact_sample_ids(artifact)
    current_ids = {record["sample_id"] for record in records}
    if not artifact_ids <= current_ids:
        mismatches.append("sample_ids")
    if mismatches:
        fields = ", ".join(sorted(set(mismatches)))
        raise ValueError(
            f"Target-shot split artifact mismatch for {fields}. "
            "Regenerate the artifact or rerun with overwrite=true after confirming the changed inputs."
        )


def load_or_build_target_shot_split(
    rows: Iterable[Mapping[str, Any] | SampleRow],
    config: TargetShotSplitConfig | Mapping[str, Any],
    *,
    dataset_type: str | None = None,
    leakage_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, TargetShotSplitConfig) else TargetShotSplitConfig.from_config(config)
    records = list(rows)
    if cfg.artifact_path and Path(cfg.artifact_path).exists() and not cfg.overwrite:
        artifact = load_target_shot_artifact(cfg.artifact_path)
        validate_target_shot_artifact(artifact, records, cfg)
        return artifact
    artifact = build_target_shot_split(records, cfg, dataset_type=dataset_type, leakage_metadata=leakage_metadata)
    if cfg.artifact_path:
        write_target_shot_artifact(artifact, cfg.artifact_path)
    return artifact


def sample_id_fingerprint(rows: Iterable[Mapping[str, Any] | SampleRow]) -> str:
    ids = [str(_row_record(row).get("sample_id", "")) for row in rows]
    payload = json.dumps(ids, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_manifest_rows(path: str | Path, *, dataset_type: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for idx, item in enumerate(csv.DictReader(handle)):
            row = dict(item)
            row.setdefault("sample_id", row.get("sample_id") or row.get("seq_index") or str(idx))
            if dataset_type:
                row.setdefault("dataset_type", dataset_type)
            rows.append(row)
    return rows


def _select_target_labeled(
    pool: list[dict[str, Any]],
    cfg: TargetShotSplitConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    count = _label_count(len(pool), cfg.target_label_fraction)
    manifest: dict[str, Any] = {
        "strategy": cfg.target_label_selection,
        "seed": int(cfg.seed),
        "target_adaptation_pool_count": int(len(pool)),
        "target_label_fraction": float(cfg.target_label_fraction),
        "selected_count": int(count),
        "buckets": {},
        "fallback_reason": None,
    }
    if count <= 0:
        return [], manifest
    rng = np.random.default_rng(int(cfg.seed))
    key_name = _strategy_field(cfg.target_label_selection)
    if cfg.target_label_selection == "random":
        selected = _random_subset(pool, count, rng)
    else:
        missing = [item for item in pool if _bucket_value(item, key_name) is None]
        if missing and len(missing) == len(pool):
            fallback = cfg.fallback_target_label_selection
            if fallback not in {"random", None}:
                raise ValueError("Only random fallback_target_label_selection is supported for target-shot splits.")
            if fallback == "random":
                manifest["fallback_reason"] = f"{key_name}_unavailable"
                selected = _random_subset(pool, count, rng)
            else:
                raise ValueError(
                    f"split.target_label_selection={cfg.target_label_selection} requires {key_name}; "
                    "set split.fallback_target_label_selection=random to allow deterministic fallback."
                )
        else:
            selected = _stratified_subset(pool, count, rng, key_name=key_name, manifest=manifest)
    manifest["selected_sample_ids"] = [item["sample_id"] for item in selected]
    return selected, manifest


def _stratified_subset(
    pool: list[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
    *,
    key_name: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        value = _bucket_value(item, key_name)
        if value is not None:
            buckets[str(value)].append(item)
    total = sum(len(items) for items in buckets.values())
    selected: list[dict[str, Any]] = []
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for bucket, items in sorted(buckets.items()):
        exact = count * (len(items) / total) if total else 0.0
        quota = min(len(items), int(np.floor(exact)))
        quotas[bucket] = quota
        remainders.append((exact - quota, bucket))
    while sum(quotas.values()) < count and remainders:
        _, bucket = max(remainders)
        remainders = [item for item in remainders if item[1] != bucket]
        if quotas[bucket] < len(buckets[bucket]):
            quotas[bucket] += 1
    for bucket, items in sorted(buckets.items()):
        chosen = _random_subset(items, quotas.get(bucket, 0), rng)
        selected.extend(chosen)
        manifest["buckets"][bucket] = {
            "candidate_count": int(len(items)),
            "selected_count": int(len(chosen)),
            "seed": int(manifest["seed"]),
            "fallback_reason": None if chosen or not items else "empty_bucket",
        }
    if len(selected) < count:
        remaining_ids = {item["sample_id"] for item in selected}
        selected.extend(_random_subset([item for item in pool if item["sample_id"] not in remaining_ids], count - len(selected), rng))
        manifest["fallback_reason"] = "stratified_quota_underfilled"
    return sorted(selected, key=lambda item: int(item["index"]))


def _random_subset(items: list[dict[str, Any]], count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    indices = np.arange(len(items))
    rng.shuffle(indices)
    selected = [items[int(idx)] for idx in sorted(indices[: min(count, len(items))])]
    return sorted(selected, key=lambda item: int(item["index"]))


def _split_target_pool(target: list[dict[str, Any]], cfg: TargetShotSplitConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explicit_test = [item for item in target if _normalize_split(item.get("split")) in {"test", "target_test", "eval", "evaluation"}]
    explicit_adapt = [item for item in target if item not in explicit_test]
    if explicit_test:
        return explicit_adapt, explicit_test
    count = int(round(len(target) * max(0.0, min(1.0, float(cfg.target_test_fraction)))))
    if len(target) > 1:
        count = min(max(count, 1), len(target) - 1)
    rng = np.random.default_rng(int(cfg.seed) + 7919)
    selected = _random_subset(target, count, rng)
    test_ids = {item["sample_id"] for item in selected}
    return [item for item in target if item["sample_id"] not in test_ids], selected


def _label_count(pool_size: int, fraction: float) -> int:
    if pool_size <= 0 or float(fraction) <= 0:
        return 0
    return min(pool_size, max(1, int(round(pool_size * float(fraction)))))


def _row_record(row: Mapping[str, Any] | SampleRow, *, index: int | None = None) -> dict[str, Any]:
    if isinstance(row, SampleRow):
        base = row.to_dict()
    else:
        base = dict(row)
    metadata = _coerce_mapping(base.get("metadata"))
    target_ref = _coerce_mapping(base.get("target_ref"))
    resource_refs = _coerce_mapping(base.get("resource_refs"))
    record = {
        **base,
        **{key: value for key, value in metadata.items() if key not in base},
        "metadata": metadata,
        "target_ref": target_ref,
        "resource_refs": resource_refs,
    }
    record["index"] = int(index if index is not None else record.get("index", record.get("row_index", 0)) or 0)
    record["sample_id"] = str(record.get("sample_id", record["index"]))
    record["split"] = str(record.get("split", metadata.get("split", "")) or "")
    return record


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _require(record: Mapping[str, Any], name: str, *, dataset_type: str, domain_type: str) -> str:
    aliases = {
        "scenario": ("scenario", "sensor_scenario", "scene_slug", "scene_id", "scene_or_city"),
        "town": ("town", "city", "city_id"),
    }[name]
    for alias in aliases:
        value = record.get(alias)
        if value not in (None, ""):
            return str(value)
    raise ValueError(
        f"Cannot build {domain_type} domain key for dataset_type={dataset_type}: missing field {name}. "
        "Add explicit metadata fields or choose a split.domain_type that only uses available fields."
    )


def _require_weather(record: Mapping[str, Any], *, dataset_type: str, domain_type: str) -> str:
    for alias in ("weather", "condition"):
        value = record.get(alias)
        if value not in (None, ""):
            return str(value)
    raise ValueError(
        f"Cannot build {domain_type} domain key for dataset_type={dataset_type}: missing field weather/condition. "
        "Add weather or condition metadata, or choose split.domain_type=scenario."
    )


def _bucket_value(item: Mapping[str, Any], key_name: str) -> Any:
    if key_name == "beam":
        for key in ("beam_abs", "beam_label", "target_beam", "label"):
            value = item.get(key)
            if value not in (None, ""):
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    value = value[0] if value else None
                return None if value is None else int(float(value))
        return None
    if key_name == "weather":
        return item.get("weather", item.get("condition"))
    if key_name == "scenario":
        return item.get("scenario", item.get("sensor_scenario", item.get("scene_slug", item.get("scene_id"))))
    return item.get(key_name)


def _strategy_field(strategy: str) -> str:
    return {
        "stratified_by_beam": "beam",
        "stratified_by_geo_sector": "geo_sector",
        "stratified_by_weather": "weather",
    }.get(strategy, "random")


def _split_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(items)),
        "sample_ids": [str(item["sample_id"]) for item in items],
        "indices": [int(item["index"]) for item in items],
        "domains": sorted({str(item.get("domain_key", "")) for item in items}),
    }


def _split_stats(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {name: _stats_for_rows(items) for name, items in splits.items()}


def _stats_for_rows(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": int(len(items)),
        "beam_histogram": _histogram(items, "beam"),
        "beam_geo_histogram": _histogram(items, "beam_geo"),
        "beam_residual_histogram": _histogram(items, "beam_residual"),
        "residual_class_histogram": _histogram(items, "residual_class"),
        "geo_sector_histogram": _histogram(items, "geo_sector"),
        "weather_histogram": _histogram(items, "weather"),
        "scenario_histogram": _histogram(items, "scenario"),
    }


def _histogram(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = []
    for item in items:
        value = _bucket_value(item, key)
        if value not in (None, ""):
            values.append(str(value))
    return {key: int(value) for key, value in sorted(Counter(values).items())}


def _leakage_diagnostics(
    splits: Mapping[str, list[dict[str, Any]]],
    *,
    leakage_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    split_sets = {name: {item["sample_id"] for item in items} for name, items in splits.items()}
    overlaps = []
    names = sorted(split_sets)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            common = sorted(split_sets[left] & split_sets[right])
            if common:
                overlaps.append({"left": left, "right": right, "sample_ids": common[:10], "count": len(common)})
    diagnostics = {
        "sample_id_overlap_count": int(sum(item["count"] for item in overlaps)),
        "sample_id_overlap_examples": overlaps[:10],
        "split_counts": {name: len(items) for name, items in split_sets.items()},
        "eligibility_reasons": [],
    }
    if leakage_metadata:
        diagnostics.update({key: value for key, value in leakage_metadata.items() if key not in diagnostics})
        if "leakage_diagnostics" in leakage_metadata and isinstance(leakage_metadata["leakage_diagnostics"], Mapping):
            diagnostics.update(dict(leakage_metadata["leakage_diagnostics"]))
    if diagnostics["sample_id_overlap_count"]:
        diagnostics["eligibility_reasons"].append("sample_id_overlap")
    if int(diagnostics.get("guard_band_violations", 0) or 0) > 0:
        diagnostics["eligibility_reasons"].append("guard_band_violation")
    return diagnostics


def _artifact_sample_ids(artifact: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for split_payload in artifact.get("splits", {}).values():
        if isinstance(split_payload, Mapping):
            ids.update(str(item) for item in split_payload.get("sample_ids", []))
    return ids


def _first(records: list[dict[str, Any]], key: str, *, default: Any = None) -> Any:
    for item in records:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_path_text(value: Any) -> str | None:
    text = _optional_text(value)
    return text if text else None


def _normalize_split(value: Any) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "ARTIFACT_VERSION",
    "SUPPORTED_DOMAIN_TYPES",
    "SUPPORTED_SELECTIONS",
    "TargetShotSplitConfig",
    "build_domain_key",
    "build_target_shot_split",
    "load_or_build_target_shot_split",
    "load_target_shot_artifact",
    "read_manifest_rows",
    "sample_id_fingerprint",
    "validate_target_shot_artifact",
    "write_target_shot_artifact",
]
