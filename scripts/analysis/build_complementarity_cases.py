#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kd_sensing.diagnostics.complementarity import (  # noqa: E402
    STRONG_MODALITIES,
    WEAK_MODALITIES,
    build_case_table,
    compute_bucket_summary,
    compute_summary,
    load_subset_predictions,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build weak-modality complementarity cases from Conditional Utility Audit outputs."
    )
    parser.add_argument("--scene", required=True, help="Scene label recorded in summary metadata.")
    parser.add_argument("--input-path", required=True, help="Audit output directory or subset_predictions table path.")
    parser.add_argument("--output-dir", required=True, help="Directory for complementarity outputs.")
    parser.add_argument("--strong-subset", default="strong_only", help="Strong-only subset name or alias.")
    parser.add_argument(
        "--strong-modalities",
        nargs="?",
        const=",".join(STRONG_MODALITIES),
        default=None,
        help=(
            "Enable strong-modality pair mode with comma-separated strong modalities. "
            "If passed without a value, defaults to gps,mmwave. Omit to keep legacy strong-only analysis."
        ),
    )
    parser.add_argument(
        "--weak-modalities",
        default=",".join(WEAK_MODALITIES),
        help="Comma-separated weak modalities. Defaults to image,radar,lidar.",
    )
    parser.add_argument(
        "--fusion-subsets",
        help=(
            "Fusion subset override. Use weak=subset pairs, e.g. "
            "image=strong_plus_image,radar=gps+mmwave+radar. A plain comma list maps by weak order."
        ),
    )
    parser.add_argument(
        "--pair-fusion-subsets",
        help=(
            "Pair fusion subset override for strong-modality mode. Use strong+weak=subset pairs, e.g. "
            "mmwave+image=mmwave_plus_image,gps+radar=gps_plus_radar."
        ),
    )
    parser.add_argument("--horizons", help="Comma-separated horizon names or indices, e.g. t+1,t+2 or 0,1.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    weak_modalities = _split_csv(args.weak_modalities) or list(WEAK_MODALITIES)
    strong_modalities = _split_csv(args.strong_modalities) if args.strong_modalities is not None else None
    fusion_subsets = _parse_fusion_subsets(args.fusion_subsets, weak_modalities)
    pair_fusion_subsets = _parse_pair_fusion_subsets(args.pair_fusion_subsets)
    horizons = _split_csv(args.horizons)

    tables = load_subset_predictions(args.input_path)
    print(f"[complementarity] input_path={args.input_path}", flush=True)
    print(f"[complementarity] loaded_tables={json.dumps(tables.paths, sort_keys=True)}", flush=True)
    print(
        "[complementarity] subset_fields="
        + ",".join(str(column) for column in tables.subset_predictions.columns),
        flush=True,
    )
    if not tables.teacher_predictions.empty:
        print(
            "[complementarity] teacher_fields="
            + ",".join(str(column) for column in tables.teacher_predictions.columns),
            flush=True,
        )

    cases, metadata = build_case_table(
        tables.subset_predictions,
        teacher_predictions=tables.teacher_predictions,
        per_sample_delta=tables.per_sample_delta,
        communication_state_features=tables.communication_state_features,
        strong_subset=args.strong_subset,
        strong_modalities=strong_modalities,
        weak_modalities=weak_modalities,
        fusion_subsets=fusion_subsets,
        pair_fusion_subsets=pair_fusion_subsets,
        horizons=horizons,
        scene=args.scene,
    )
    bucket_summary, bucket_metadata = compute_bucket_summary(
        cases,
        tables.communication_state_features,
        return_metadata=True,
    )
    metadata["bucket"] = bucket_metadata
    metadata["input_tables"] = tables.paths
    summary = compute_summary(cases, metadata=metadata, scene=args.scene)
    outputs = write_outputs(cases, summary, bucket_summary, args.output_dir)

    print(
        "[complementarity] schema_mapping="
        + json.dumps(metadata.get("schema", {}).get("subset_predictions", {}).get("field_mapping", {}), sort_keys=True),
        flush=True,
    )
    print(
        "[complementarity] weak_prediction_sources="
        + json.dumps(metadata.get("weak_prediction_sources", {}), sort_keys=True),
        flush=True,
    )
    if metadata.get("analysis_mode") == "strong_modality_pair":
        print(
            "[complementarity] strong_prediction_sources="
            + json.dumps(metadata.get("strong_prediction_sources", {}), sort_keys=True),
            flush=True,
        )
        print(
            "[complementarity] fusion_subset_availability="
            + json.dumps(metadata.get("fusion_subset_availability", {}), sort_keys=True),
            flush=True,
        )
        print(
            "[complementarity] unmatched_pairs="
            + json.dumps(metadata.get("unmatched", {}), sort_keys=True),
            flush=True,
        )
    print(
        "[complementarity] probability_metrics_available="
        + str(bool(metadata.get("probability_metrics_available"))),
        flush=True,
    )
    if metadata.get("warnings"):
        for warning in metadata["warnings"]:
            print(f"[complementarity] warning={warning}", flush=True)
    if bucket_metadata.get("bucket_statistics_unavailable_reason"):
        print(
            "[complementarity] warning="
            + str(bucket_metadata["bucket_statistics_unavailable_reason"]),
            flush=True,
        )
    print(f"[complementarity] output_rows={len(cases)}", flush=True)
    print("[complementarity] outputs=" + json.dumps(outputs, sort_keys=True), flush=True)
    return {
        "status": "complete",
        "scene": args.scene,
        "case_rows": int(len(cases)),
        "outputs": outputs,
        "summary": summary,
    }


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_fusion_subsets(value: str | None, weak_modalities: list[str]) -> dict[str, str]:
    items = _split_csv(value)
    if not items:
        return {}
    if any("=" in item for item in items):
        result: dict[str, str] = {}
        for item in items:
            if "=" not in item:
                continue
            weak, subset = item.split("=", 1)
            if weak.strip() and subset.strip():
                result[weak.strip()] = subset.strip()
        return result
    return {weak: subset for weak, subset in zip(weak_modalities, items)}


def _parse_pair_fusion_subsets(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _split_csv(value):
        if "=" not in item:
            continue
        pair, subset = item.split("=", 1)
        if pair.strip() and subset.strip():
            result[pair.strip()] = subset.strip()
    return result


if __name__ == "__main__":
    main()
