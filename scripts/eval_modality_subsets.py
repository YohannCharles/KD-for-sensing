#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.data_factory import build_dataloaders  # noqa: E402
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion  # noqa: E402
from kd_sensing.engine.validator import validate  # noqa: E402
from kd_sensing.utils.checkpoint import load_model_state  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate prior-driven modality subsets for a fusion checkpoint.")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--ckpt", "--weights", dest="ckpt", required=True)
    parser.add_argument("--subsets", nargs="*", help="Subset names to evaluate.")
    parser.add_argument("--override", "-o", action="append", default=[])
    args, unknown = parser.parse_known_args(argv)
    cfg = load_config(args.config, [*args.override, *(item for item in unknown if "=" in item)])
    subset_cfg = cfg.setdefault("evaluation", {}).setdefault("modality_subsets", {})
    subset_cfg["enabled"] = True
    if args.subsets:
        subset_cfg["subsets"] = list(args.subsets)
    device = build_device(cfg)
    model = build_model(cfg["model"]["student"]).to(device)
    load_model_state(args.ckpt, model, role="eval_modality_subsets", map_location=device, strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)))
    criterion = build_task_criterion(cfg)
    metrics = validate(model, build_dataloaders(cfg)["test"], cfg, criterion, device)
    result = {"modality_subsets": metrics.get("modality_subsets", {})}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
