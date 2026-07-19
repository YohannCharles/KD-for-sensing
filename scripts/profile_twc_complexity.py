#!/usr/bin/env python3
import argparse
import json
import platform
from pathlib import Path

import torch

from kd_sensing.config.io import load_config
from kd_sensing.engine.optim import build_model
from kd_sensing.evaluation.complexity import benchmark_forward, estimate_macs, parameter_summary
from kd_sensing.utils.checkpoint import load_model_state


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one TWC model with a fixed synthetic four-modality input.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--amp", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(",") if value.strip())
    if any(value <= 0 for value in batch_sizes):
        parser.error("batch sizes must be positive")
    cfg = load_config(_path(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("TWC complexity profiling requires CUDA for paper latency and memory evidence.")
    device = torch.device("cuda")
    model = build_model(cfg["model"]["primary"]).to(device).eval()
    checkpoint = _path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None:
        load_model_state(checkpoint, model, role="TWC complexity profile", map_location=device, strict=True)
    amp_enabled = args.amp == "fp16"
    measurements = []
    macs = None
    for batch_size in batch_sizes:
        inputs = _inputs(batch_size, cfg, device)

        def forward():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                return model(**inputs)

        if batch_size == 1:
            macs = estimate_macs(forward, device=device)
        measurements.append(
            {
                "batch_size": batch_size,
                **benchmark_forward(
                    forward,
                    device=device,
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repeats=args.repeats,
                ),
            }
        )
        del inputs
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "artifact_kind": "twc_complexity_profile_v1",
        "config": str(_path(args.config)),
        "checkpoint": str(checkpoint) if checkpoint else None,
        **parameter_summary(model),
        "macs_batch1": macs,
        "measurements": measurements,
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "python_platform": platform.platform(),
        },
        "policy": {
            "amp": args.amp,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "history_window": int(cfg["model"].get("seq_length", 5)),
            "image_size": list(cfg["data"]["dataset"].get("image_size", (224, 224))),
        },
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))
    return 0


def _inputs(batch_size: int, cfg: dict, device: torch.device) -> dict[str, torch.Tensor]:
    steps = int(cfg["model"].get("seq_length", 5))
    height, width = cfg["data"]["dataset"].get("image_size", (224, 224))
    return {
        "image_batch": torch.randn(batch_size, steps, 3, height, width, device=device),
        "radar_batch": torch.randn(batch_size, steps, 2, 128, 64, device=device),
        "gps_batch": torch.randn(batch_size, steps, 3, device=device),
        "lidar_batch": torch.randn(batch_size, steps, 3, height, width, device=device),
        "modality_temporal_mask": torch.ones(batch_size, steps, 4, dtype=torch.bool, device=device),
    }


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

