#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import generate_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import (
    frequency_offsets_hz,
    load_path_channel,
    pilot_subcarrier_indices,
    simulate_candidate_pilots,
)
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.config.io import dump_config
from kd_sensing.data.mmw.pilot_alignment import resolve_last_input_channel_ref


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.split_role != "train":
        raise ValueError("Codebook/cache creation is train-only; validation/test may only load published artifacts.")
    config = safe_load_yaml(args.config.read_text(encoding="utf-8"))
    channel_cfg = config["channel"]
    codebook_cfg = config["pilot_codebook"]
    indices = pilot_subcarrier_indices(
        channel_cfg["num_subcarriers"],
        channel_cfg["pilot_subcarriers"],
        pattern=channel_cfg["pilot_frequency_pattern"],
    )
    frequencies = frequency_offsets_hz(
        indices,
        num_subcarriers=channel_cfg["num_subcarriers"],
        subcarrier_spacing_hz=channel_cfg["subcarrier_spacing_hz"],
        mode=channel_cfg["frequency_index_mode"],
    )
    codebook = generate_probe_codebook(
        64,
        16,
        num_patterns=codebook_cfg["num_candidate_patterns"],
        seed=codebook_cfg["seed"],
        method=codebook_cfg["method"],
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_root / "config_resolved.yaml")
    codebook_path = codebook.save(args.output_root / "pilot_codebook.npz")
    cache = PilotCache(args.cache_root)
    spec = PilotCacheSpec(
        codebook.hash,
        tuple(float(value) for value in frequencies),
        float(channel_cfg["subcarrier_spacing_hz"]),
        str(channel_cfg["frequency_index_mode"]),
        64,
        16,
    )
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = rows[: min(len(rows), int(args.limit))]
    entries = []
    for row in selected_rows:
        csi_columns = sorted(
            (key for key in row if key.startswith("csi") and key[3:].isdigit()), key=lambda key: int(key[3:])
        )
        reference = resolve_last_input_channel_ref(
            row,
            [row[key] for key in csi_columns],
            data_root=args.dataset_root / row["condition"],
            seq_len=len(csi_columns),
            num_pred=1,
        )
        channel_path = Path(reference["channel_ref"])

        def compute(path=channel_path):
            matrices, delays = load_path_channel(path)
            wrapped = matrices[None, None, :, None, :, :, None]
            wrapped_tau = delays[None, None, None, :]
            return simulate_candidate_pilots(wrapped, wrapped_tau, codebook, frequencies)

        values = cache.get_or_compute(channel_path, spec, compute)
        entries.append({"sample_id": row.get("sample_id"), "channel_ref": str(channel_path), "shape": list(values.shape)})
    manifest = {
        "status": "passed",
        "split_role": "train",
        "sample_count": len(entries),
        "candidate_shape": [codebook_cfg["num_candidate_patterns"], channel_cfg["pilot_subcarriers"]],
        "codebook_path": str(codebook_path.resolve()),
        "codebook_hash": codebook.hash,
        "frequency_indices": indices.tolist(),
        "frequency_positions_hz": frequencies.tolist(),
        "cache_root": str(args.cache_root.resolve()),
        "outer_test_accessed": False,
        "entries": entries,
    }
    (args.output_root / "pilot_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only noiseless sparse-pilot cache.")
    parser.add_argument("--config", type=Path, default=Path("tools/configs/sparse_pilot_transition.yaml"))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/MMW"))
    parser.add_argument("--split-role", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sparse_pilot_transition"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/sparse_pilot_transition"))
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "candidate_shape", "codebook_hash")}, indent=2))


if __name__ == "__main__":
    main()
