"""Validate the only supported MMW data protocol."""

from __future__ import annotations

from typing import Any, Mapping

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
    if mode == TRAJECTORY_PROTOCOL_MODE:
        return validate_trajectory_config_protocol(cfg)
    raise ValueError(f"MMW training requires data_protocol.mode={TRAJECTORY_PROTOCOL_MODE!r}.")


__all__ = ["validate_mmw_config_protocol"]
