from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from kd_sensing.data.mmw_town_gps_lidar_bgam_manifest import build_mmw_town_gps_lidar_bgam_manifest
from kd_sensing.engine.deepsense6g_gps_lidar_bgam import run_deepsense6g_gps_lidar_bgam


def run_mmw_town_gps_lidar_bgam(
    cfg: Mapping[str, Any],
    *,
    label_space: str | None = None,
    topk: int | None = None,
    bgam_mode: str | None = None,
    output_dir: str | Path | None = None,
    ckpt: str | Path | None = None,
    evaluate_only: bool = False,
    debug_masks: bool | None = None,
) -> dict[str, Any]:
    build_mmw_town_gps_lidar_bgam_manifest(
        cfg,
        label_space=label_space,
        topk=topk,
        output_dir=output_dir,
    )
    return run_deepsense6g_gps_lidar_bgam(
        cfg,
        support_ratio=None,
        label_space=label_space,
        topk=topk,
        bgam_mode=bgam_mode,
        output_dir=output_dir,
        ckpt=ckpt,
        evaluate_only=evaluate_only,
        debug_masks=debug_masks,
    )


def evaluate_mmw_town_gps_lidar_bgam(
    cfg: Mapping[str, Any],
    *,
    ckpt: str | Path | None = None,
    output_dir: str | Path | None = None,
    label_space: str | None = None,
    topk: int | None = None,
    bgam_mode: str | None = None,
    debug_masks: bool | None = None,
) -> dict[str, Any]:
    return run_mmw_town_gps_lidar_bgam(
        cfg,
        label_space=label_space,
        topk=topk,
        bgam_mode=bgam_mode,
        output_dir=output_dir,
        ckpt=ckpt,
        evaluate_only=True,
        debug_masks=debug_masks,
    )


__all__ = [
    "evaluate_mmw_town_gps_lidar_bgam",
    "run_mmw_town_gps_lidar_bgam",
]
