#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_h5_p1_temporal_models_v1 as base


DEFAULT_OUTPUT_ROOT = "outputs/temporal_router_s1_s4_v1"
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


def method_specs() -> dict[str, dict[str, Any]]:
    temporal_only = {
        "difficulty": {"enabled": False, "profiles": []},
        "random_modality_dropout": {"enabled": False},
        "training": {
            "mask_sampler": "default",
            "missing_pattern_sampler": "default",
            "random_modality_dropout": {"enabled": False},
        },
        "loss": {
            "u_mask_beam_jepa": {
                "missing_pattern_sampler": "default",
                "missing_pattern": {"available_modalities": list(base.MODALITIES)},
            }
        },
        "model": {"primary": {"mask_sampler": "temporal_missing"}},
    }
    proto = {
        "model": {
            "primary": {
                "fusion_type": "supervised_router",
                "head_type": "prototype",
                "use_beam_prototype_alignment": True,
                "router_supervision": "oracle",
                "router_distill_weight": 0.1,
                "temporal_router_distill_weight": 0.1,
                "router_fuse_level": "logits",
                "temporal_aggregation": "masked_mean",
                "missing_modality_metadata": {"enabled": True, "strict": False},
            }
        },
        "training": {
            "use_beam_prototype_alignment": True,
            "use_modality_prototype_loss": True,
            "use_circular_soft_targets": True,
            "router_supervision": "oracle",
            "router_distill_weight": 0.1,
            "temporal_router_distill_weight": 0.1,
        },
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "router_supervision": "oracle",
                "router_distill_weight": 0.1,
                "temporal_router_distill_weight": 0.1,
            },
            "hard_subset_weighting": {"enabled": True, "mode": "soft_static"},
        },
        "checkpoint": {"selection_metric": "avg_missing_top1"},
    }
    temporal = {
        "temporal_missing": {
            "enabled": True,
            "history_window": 5,
            "prediction_window": 1,
            "mode": "stratified_modality_temporal",
            "mask_sampler": "stratified_modality_temporal",
            "train_missing_drop_counts": "0,1,2,3",
            "train_temporal_missing_rates": "0.0,0.2,0.4,0.6,0.8",
            "train_temporal_missing_types": "modality_level,frame_level,modality_frame,block",
            "ensure_at_least_one_cell": True,
            "ensure_at_least_one_frame": True,
            "ensure_at_least_one_modality": True,
            "seed": 0,
        }
    }
    baseline_mask = {
        "model": {"primary": {"missing_modality_metadata": {"enabled": True, "strict": False}}},
        "temporal_missing": temporal["temporal_missing"],
    }
    return {
        "s1_temporalagg_modality_router": {
            "base_config": base.DEFAULT_C2_CONFIG,
            "overrides": base._merge(proto, temporal_only, temporal, {"model": {"primary": {"temporal_router_type": "s1_temporalagg_modality"}}}),
            "mapping": "u_mask_beam_jepa S1 TemporalAgg -> supervised modality router",
        },
        "s2_pertime_modality_router": {
            "base_config": base.DEFAULT_C2_CONFIG,
            "overrides": base._merge(proto, temporal_only, temporal, {"model": {"primary": {"temporal_router_type": "s2_pertime_modality"}}}),
            "mapping": "u_mask_beam_jepa S2 per-time supervised modality router + masked temporal mean",
        },
        "s3_two_level_router": {
            "base_config": base.DEFAULT_C2_CONFIG,
            "overrides": base._merge(proto, temporal_only, temporal, {"model": {"primary": {"temporal_router_type": "s3_two_level"}}}),
            "mapping": "u_mask_beam_jepa S3 per-time modality router + temporal router",
        },
        "s4_global_modality_time_router": {
            "base_config": base.DEFAULT_C2_CONFIG,
            "overrides": base._merge(proto, temporal_only, temporal, {"model": {"primary": {"temporal_router_type": "s4_global"}}}),
            "mapping": "u_mask_beam_jepa S4 global modality-time router over 5x4 cells",
        },
        "amber_full": {
            "base_config": base.DEFAULT_AMBER_CONFIG,
            "overrides": base._merge(temporal_only, baseline_mask, {"model": {"primary": {"encoders": {"image": {"freeze_backbone": False}, "radar": {"freeze_backbone": False}, "lidar": {"freeze_backbone": False}}}}}),
            "mapping": "AMBER Full with zero-filled [B,5,4] temporal-modality mask metadata",
        },
        "rmbp_mm": {
            "base_config": base.DEFAULT_RMBP_CONFIG,
            "overrides": base._merge(temporal_only, baseline_mask, {"model": {"primary": {"encoders": {"image": {"freeze_backbone": False, "unfreeze_stages": []}}}}}),
            "mapping": "RMBP-MM channel-attention baseline with temporal-modality mask metadata",
        },
    }


_BASE_PLAN_JOBS = base.plan_jobs


def plan_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.methods = ",".join(ALIASES.get(item, item) for item in base._csv(args.methods))
    return _BASE_PLAN_JOBS(args)


def main(argv: list[str] | None = None) -> int:
    argv = _with_temporal_router_defaults(list(sys.argv[1:] if argv is None else argv))
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_METHODS = DEFAULT_METHODS
    base.method_specs = method_specs
    base.plan_jobs = plan_jobs
    return base.main(argv)


def _with_temporal_router_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    defaults = {
        "--gpus": "2,3,4,5,6,7",
        "--max_jobs": "6",
        "--umask_batch_size": "64",
        "--baseline_batch_size": "64",
        "--num_workers": "4",
        "--prefetch_factor": "2",
        "--torch_num_threads": "4",
        "--torch_num_interop_threads": "2",
    }
    for flag, value in defaults.items():
        dashed = flag.replace("_", "-")
        if flag not in result and dashed not in result:
            result.extend([flag, value])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
