#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from kd_sensing.data.deepsense_twc import load_protocol, prepare_protocol


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the independent DeepSense6G TWC secondary protocol.")
    parser.add_argument("--dataset-root", default="dataset/DeepSense6G")
    parser.add_argument("--cache-root", default="outputs/cache/deepsense6g_twc_secondary_v1")
    args = parser.parse_args()
    path = prepare_protocol(_path(args.dataset_root), _path(args.cache_root))
    protocol = load_protocol(path)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_sha256": protocol["manifest_sha256"],
                "scenes": len(protocol["scenes"]),
                "dataset_scope": protocol["pooled_dataset"]["id"],
                "pooled_train_samples": protocol["pooled_dataset"]["train_row_count"],
                "pooled_test_samples": protocol["pooled_dataset"]["test_row_count"],
                "fixed_conditions": protocol["fixed_mask_cache"]["condition_count"],
            },
            indent=2,
        )
    )
    return 0


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
