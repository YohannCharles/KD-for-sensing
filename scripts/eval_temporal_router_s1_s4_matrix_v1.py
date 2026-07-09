#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_h5_p1_temporal_matrix_v1 as base


DEFAULT_METHODS = (
    "s1_temporalagg_modality_router,"
    "s2_pertime_modality_router,"
    "s3_two_level_router,"
    "s4_global_modality_time_router,"
    "amber_full,"
    "rmbp_mm"
)
ALIASES = {
    "s1": "s1_temporalagg_modality_router",
    "s2": "s2_pertime_modality_router",
    "s3": "s3_two_level_router",
    "s4": "s4_global_modality_time_router",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args = _default(args, "--root", "outputs/temporal_router_s1_s4_v1")
    args = _default(args, "--methods", DEFAULT_METHODS)
    args = _default(args, "--eval_fixed_mask_cache", "outputs/temporal_eval_masks_s1_s4_v1")
    args = _default(args, "--output_dir", "outputs/temporal_router_s1_s4_v1/eval_matrix")
    if "--methods" in args:
        index = args.index("--methods") + 1
        args[index] = ",".join(ALIASES.get(item.strip(), item.strip()) for item in args[index].split(",") if item.strip())
    return base.main(args)


def _default(args: list[str], flag: str, value: str) -> list[str]:
    dashed = flag.replace("_", "-")
    if flag in args or dashed in args:
        return args
    return [flag, value, *args]


if __name__ == "__main__":
    raise SystemExit(main())
