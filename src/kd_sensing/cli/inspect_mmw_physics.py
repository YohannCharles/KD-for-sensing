import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config import load_config
from kd_sensing.data.datasets.mmw_physics_adapter import build_mmw_physics_targets, physics_shape_summary
from kd_sensing.registries import DATASETS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect MMW physics supervision shapes from a package CLI.")
    parser.add_argument("--config", type=Path, default=Path("configs/fusion/physics_informed_mmw_debug.yaml"))
    parser.add_argument("--split", choices=("train", "test", "val"), default="train")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("-o", "--override", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.override)
    dataset_cfg = dict(cfg.get("data", {}).get("dataset", {}))
    dataset_cfg["physics_supervision"] = dataset_cfg.get("physics_supervision", True)
    if args.split == "test" and dataset_cfg.get("test_csv_name"):
        dataset_cfg["csv_name"] = dataset_cfg["test_csv_name"]
    if args.split == "val" and dataset_cfg.get("val_csv_name"):
        dataset_cfg["csv_name"] = dataset_cfg["val_csv_name"]
    import_default_components()
    dataset = DATASETS.build(dataset_cfg)
    rows: list[dict[str, Any]] = []
    for index in range(min(max(int(args.max_samples), 0), len(dataset))):
        sample = dataset[index]
        sample["physics_targets"] = build_mmw_physics_targets(sample, dataset_cfg.get("physics_supervision"))
        summary = physics_shape_summary(sample)
        summary["index"] = index
        rows.append(summary)
    print(json.dumps({"config": str(args.config), "split": args.split, "samples": rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
