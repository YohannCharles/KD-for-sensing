from __future__ import annotations

from pathlib import Path
from typing import Any

from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.preparation_audit import _extract_zip, validate_zip_inputs, write_data_availability
from kd_sensing.data.mmw.preparation_beam_power import derive_beam_power, derive_beam_power_from_file, load_channel_payload
from kd_sensing.data.mmw.preparation_config import (
    ALGORITHM_VERSION,
    DEFAULT_SCENARIO,
    DEFAULT_TOWN,
    GROUP_SAFE_TIME_BLOCK,
    MMWPreparationConfig,
    MMW_SPLIT_PROTOCOL_VERSION,
    SUPPORTED_SEQUENCE_SPLIT_STRATEGIES,
    load_preparation_config,
)
from kd_sensing.data.mmw.preparation_geometry import build_proxy_features, build_relative_geometry
from kd_sensing.data.mmw.preparation_index import (
    ChannelFile,
    PreparedFrame,
    SensorFrame,
    default_channel_scenario,
    index_channel_files,
    index_sensor_frames,
    sample_id_for,
)
from kd_sensing.data.mmw.preparation_splits import (
    build_sequence_rows,
    build_sequence_splits_from_manifest,
    compute_split_leakage_diagnostics,
    split_sequence_rows,
)
from kd_sensing.data.mmw.preparation_writers import build_prepared_artifacts

def prepare_town10_skybridge(
    config: MMWPreparationConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    layout = mmw_condition_layout(config.condition)
    if dry_run and (not config.sensor_zip.exists() or not config.channel_zip.exists()):
        return {
            "status": "dry_run",
            "condition_root": str(config.condition_root),
            "layout": {
                "sensor_data_root": str(Path(config.output_root) / Path(layout.sensor_data_root).relative_to("dataset")),
                "channel_data_root": str(Path(config.output_root) / Path(layout.channel_data_root).relative_to("dataset")),
                "prepared_root": str(config.prepared_root),
            },
            "missing_zips": [
                str(path.resolve())
                for path in (config.sensor_zip, config.channel_zip)
                if not path.exists()
            ],
            "message": "Dry-run skipped extraction and indexing because one or more zip files are not present.",
        }

    zip_info = validate_zip_inputs(config)
    if dry_run:
        return {
            "status": "dry_run",
            "condition_root": str(config.condition_root),
            "sensor_root": str(config.sensor_root),
            "channel_root": str(config.channel_root),
            "prepared_root": str(config.prepared_root),
            "zip_inputs": zip_info,
            "message": "Dry-run validated zip paths and resolved output layout; extraction and artifact writes were skipped.",
        }
    if not dry_run:
        _extract_zip(config.sensor_zip, config.sensor_root, force=force)
        _extract_zip(config.channel_zip, config.channel_root, force=force)

    sensor_index = index_sensor_frames(config.sensor_root, town=config.town, scenario=config.scenario)
    channel_index = index_channel_files(
        config.channel_root,
        town=config.town,
        scenario=config.scenario,
        channel_scenario=config.resolved_channel_scenario,
    )
    result = build_prepared_artifacts(config, sensor_index, channel_index, zip_info=zip_info, dry_run=dry_run)
    if not dry_run:
        availability = write_data_availability(config.condition_root)
        result["data_availability_path"] = availability["path"]
    return result

__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_SCENARIO",
    "DEFAULT_TOWN",
    "GROUP_SAFE_TIME_BLOCK",
    "MMWPreparationConfig",
    "MMW_SPLIT_PROTOCOL_VERSION",
    "SUPPORTED_SEQUENCE_SPLIT_STRATEGIES",
    "ChannelFile",
    "PreparedFrame",
    "SensorFrame",
    "build_prepared_artifacts",
    "build_proxy_features",
    "build_relative_geometry",
    "build_sequence_rows",
    "build_sequence_splits_from_manifest",
    "compute_split_leakage_diagnostics",
    "default_channel_scenario",
    "derive_beam_power",
    "derive_beam_power_from_file",
    "index_channel_files",
    "index_sensor_frames",
    "load_channel_payload",
    "load_preparation_config",
    "prepare_town10_skybridge",
    "sample_id_for",
    "split_sequence_rows",
    "validate_zip_inputs",
    "write_data_availability",
]
