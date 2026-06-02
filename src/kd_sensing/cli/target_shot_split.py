from __future__ import annotations

import argparse
import json
from pathlib import Path

from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.target_shot_splits import (
    TargetShotSplitConfig,
    build_target_shot_split,
    read_manifest_rows,
    write_target_shot_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a target-shot source/target split artifact.")
    parser.add_argument("--config", "-c", required=True, help="YAML config containing a split section.")
    parser.add_argument("--input", required=False, help="Manifest or sequence CSV. Defaults to split.input_path.")
    parser.add_argument("--output", required=False, help="Output JSON/NPZ artifact path. Defaults to split.artifact_path.")
    parser.add_argument("--dataset-type", default=None, help="Dataset type for diagnostics and domain error messages.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output artifact.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = safe_load_yaml(Path(args.config).read_text(encoding="utf-8")) or {}
    split_cfg = dict(cfg.get("split", {}))
    input_path = args.input or split_cfg.get("input_path") or cfg.get("input_path")
    output_path = args.output or split_cfg.get("artifact_path") or split_cfg.get("output_path")
    if not input_path:
        parser.error("--input or split.input_path is required.")
    if not output_path:
        parser.error("--output or split.artifact_path is required.")
    if Path(output_path).exists() and not (args.overwrite or bool(split_cfg.get("overwrite", False))):
        raise FileExistsError(f"Target-shot split artifact exists: {output_path}. Use --overwrite to replace it.")
    split_cfg["artifact_path"] = str(output_path)
    split_cfg["overwrite"] = bool(args.overwrite or split_cfg.get("overwrite", False))
    config = TargetShotSplitConfig.from_config({"split": split_cfg})
    dataset_type = args.dataset_type or cfg.get("data", {}).get("dataset", {}).get("type") or cfg.get("dataset_type")
    rows = read_manifest_rows(input_path, dataset_type=dataset_type)
    artifact = build_target_shot_split(rows, config, dataset_type=dataset_type)
    outputs = write_target_shot_artifact(artifact, output_path)
    print(json.dumps({"artifact": outputs["json"], "npz": outputs["npz"], "counts": artifact["stats"]}, indent=2))


if __name__ == "__main__":
    main()
