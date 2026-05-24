from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kd_sensing.data.layouts import MULTIMODAL_NF_FAMILY
from kd_sensing.preprocessing.multimodal_nf_codebook import _fingerprint, fingerprint_path, parse_codebook_metadata
from kd_sensing.preprocessing.multimodal_nf_constants import MULTIMODAL_NF_DATASET_TYPE, REQUIRED_MULTIMODAL_NF_FIELDS
from kd_sensing.preprocessing.multimodal_nf_hdf5 import _candidate_codebook_files, _candidate_hdf5_files, _hdf5_file_summary
from kd_sensing.preprocessing.multimodal_nf_paths import resolve_multimodal_nf_paths

def audit_multimodal_nf_files(
    data_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    codebook_root: str | Path | None = None,
    channel_path: str | Path | None = None,
    image_path: str | Path | None = None,
    lidar_path: str | Path | None = None,
    codebook_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    paths = resolve_multimodal_nf_paths(
        data_root=data_root,
        raw_root=raw_root,
        codebook_root=codebook_root,
        output_dir=output_dir,
    )
    hdf5_files = _candidate_hdf5_files(paths, channel_path=channel_path, image_path=image_path, lidar_path=lidar_path)
    summaries = [_hdf5_file_summary(path) for path in hdf5_files]
    combined_keys = {field for summary in summaries for field in summary.get("resolved_fields", {})}
    missing = [field for field in REQUIRED_MULTIMODAL_NF_FIELDS if field not in combined_keys]
    codebooks = _candidate_codebook_files(paths, codebook_path=codebook_path)
    codebook_summaries = []
    for item in codebooks:
        try:
            codebook_summaries.append(parse_codebook_metadata(item))
        except Exception as exc:
            codebook_summaries.append({"path": str(item), "error": str(exc), "fingerprint": fingerprint_path(item)})
    cities = sorted({city for summary in summaries for city in summary.get("cities", [])})
    sample_count = int(sum(int(summary.get("sample_count", 0) or 0) for summary in summaries))
    report = {
        "dataset": MULTIMODAL_NF_DATASET_TYPE,
        "family": MULTIMODAL_NF_FAMILY,
        "data_root": str(paths.data_root),
        "raw_root": str(paths.raw_root),
        "hdf5_files": summaries,
        "codebooks": codebook_summaries,
        "city_ids": cities,
        "sample_count": sample_count,
        "missing_fields": missing,
        "fingerprint": _fingerprint([json.dumps(summary, sort_keys=True) for summary in summaries]),
    }
    if require_complete and missing:
        raise ValueError(
            "Multimodal-NF audit found missing required field(s): "
            f"{missing}. Files checked: {[str(path) for path in hdf5_files]}"
        )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.output_dir / "multimodal_nf_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report

__all__ = ["audit_multimodal_nf_files"]
