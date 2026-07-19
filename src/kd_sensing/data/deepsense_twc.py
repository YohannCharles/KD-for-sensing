"""Frozen DeepSense6G secondary-evidence protocol helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from kd_sensing.data.mmw.twc_evidence import build_fixed_mask_cache


PROTOCOL_ID = "deepsense6g_twc_secondary_v1"
SCENES = (31, 32, 33, 34)
MASK_SEED = 20260718
GENERATOR_REVISION = 3


def prepare_protocol(dataset_root: Path, cache_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    cache_root = cache_root.resolve()
    manifest_path = cache_root / "protocol_manifest.json"
    request = {
        "protocol_id": PROTOCOL_ID,
        "dataset_root": str(dataset_root),
        "scenes": list(SCENES),
        "mask_seed": MASK_SEED,
        "generator_revision": GENERATOR_REVISION,
    }
    request_sha256 = sha256_payload(request)
    if manifest_path.exists():
        existing = load_protocol(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing DeepSense6G protocol differs from the immutable request: {manifest_path}")
        return manifest_path

    cache_root.mkdir(parents=True, exist_ok=True)
    masks = build_fixed_mask_cache(seed=MASK_SEED)
    masks.update(protocol_id=PROTOCOL_ID, generator="deepsense6g_twc_fixed_mask_v1")
    masks["checksum"] = sha256_payload({key: value for key, value in masks.items() if key != "checksum"})
    mask_path = cache_root / "fixed_mask_cache.json"
    write_json(mask_path, masks)
    scene_records = []
    for scene in SCENES:
        root = dataset_root / f"scenario{scene}"
        train = root / "train_seqs_RA_GPS_LIDAR.csv"
        test = root / "test_seqs_RA_GPS_LIDAR.csv"
        for path in (train, test):
            if not path.is_file():
                raise FileNotFoundError(f"DeepSense6G protocol input is missing: {path}")
        split_dir = cache_root / "splits" / f"scene{scene}"
        scene_records.append(
            {
                "scene": int(scene),
                "data_root": str(root),
                "train": _filter_valid_future_beam_rows(train, split_dir / "train.csv", root),
                "test": _filter_valid_future_beam_rows(test, split_dir / "test.csv", root),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_kind": "independent_secondary_cross_scenario_evidence",
        "request": request,
        "request_sha256": request_sha256,
        "scenes": scene_records,
        "pooled_dataset": {
            "id": "deepsense6g_scene31_34_pooled_v1",
            "scene_ids": list(SCENES),
            "train_row_count": sum(int(item["train"]["row_count"]) for item in scene_records),
            "test_row_count": sum(int(item["test"]["row_count"]) for item in scene_records),
            "component_inventory_sha256": sha256_payload(
                {
                    "scenes": [
                        {
                            "scene": item["scene"],
                            "train_sha256": item["train"]["sha256"],
                            "test_sha256": item["test"]["sha256"],
                        }
                        for item in scene_records
                    ]
                }
            ),
        },
        "fixed_mask_cache": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "checksum": masks["checksum"],
            "condition_count": len(masks["conditions"]),
        },
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    write_json(manifest_path, payload)
    return manifest_path


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get("manifest_sha256", ""))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("protocol_id") != PROTOCOL_ID or recorded != sha256_payload(body):
        raise ValueError(f"Invalid or drifted DeepSense6G TWC protocol manifest: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _csv_record(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"DeepSense6G protocol CSV is empty: {path}") from exc
        count = sum(1 for _ in reader)
    if count <= 0:
        raise ValueError(f"DeepSense6G protocol CSV has no samples: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "row_count": count, "columns": len(header)}


def _filter_valid_future_beam_rows(source: Path, target: Path, data_root: Path) -> dict[str, Any]:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if "future_beam1" not in fieldnames:
        raise ValueError(f"DeepSense6G protocol CSV lacks future_beam1: {source}")
    accepted = []
    rejected = []
    for index, row in enumerate(rows):
        power_path = (data_root / str(row["future_beam1"])).resolve()
        try:
            powers = np.asarray(np.loadtxt(power_path, dtype=np.float64)).reshape(-1)
            valid = powers.size == 64 and bool(np.isfinite(powers).all()) and bool((powers >= 0).all())
        except Exception:
            valid = False
        if valid:
            accepted.append(row)
        else:
            rejected.append({"row_index": index, "future_beam1": str(row["future_beam1"])})
    if not accepted:
        raise ValueError(f"DeepSense6G protocol filtering removed every row: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)
    return {
        **_csv_record(target),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_row_count": len(rows),
        "invalid_future_beam1_count": len(rejected),
        "invalid_rows_sha256": sha256_payload({"rows": rejected}),
        "filter_rule": "future_beam1_has_exactly_64_finite_nonnegative_values",
    }


__all__ = ["MASK_SEED", "PROTOCOL_ID", "SCENES", "load_protocol", "prepare_protocol"]
