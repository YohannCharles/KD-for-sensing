"""Dispatch the explicit MMW development protocol without weakening either mode."""

from __future__ import annotations

from typing import Any, Mapping

from kd_sensing.data.mmw.clean_protocol import CLEAN_PROTOCOL_MODE, validate_clean_config_protocol
from kd_sensing.data.mmw.full_pool_protocol import FULL_POOL_PROTOCOL_MODE, validate_full_pool_config_protocol
from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_PROTOCOL_MODE,
    validate_trajectory_config_protocol,
)


def validate_mmw_config_protocol(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    dataset = cfg.get("data", {}).get("dataset", {})
    if str(dataset.get("type", "")).strip().lower() != "mmw":
        return None
    section = cfg.get("data_protocol")
    mode = section.get("mode") if isinstance(section, Mapping) else None
    if mode == CLEAN_PROTOCOL_MODE:
        return validate_clean_config_protocol(cfg)
    if mode == FULL_POOL_PROTOCOL_MODE:
        return validate_full_pool_config_protocol(cfg)
    if mode == TRAJECTORY_PROTOCOL_MODE:
        return validate_trajectory_config_protocol(cfg)
    raise ValueError(
        "MMW training requires an explicit supported protocol: "
        f"{CLEAN_PROTOCOL_MODE!r}, {FULL_POOL_PROTOCOL_MODE!r}, or {TRAJECTORY_PROTOCOL_MODE!r}."
    )


__all__ = ["validate_mmw_config_protocol"]
