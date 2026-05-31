from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kd_sensing.config.io import deep_merge, parse_overrides, safe_load_yaml

DEFAULT_TOWN = "Town10"
DEFAULT_SCENARIO = "Town10_skybridge_seed24"
ALGORITHM_VERSION = "mmw_channel_to_dft_power_v1"
MMW_SPLIT_PROTOCOL_VERSION = "mmw_sequence_split_v2"
GROUP_SAFE_TIME_BLOCK = "group_safe_time_block"
SUPPORTED_SEQUENCE_SPLIT_STRATEGIES = {GROUP_SAFE_TIME_BLOCK}

@dataclass(frozen=True)
class MMWPreparationConfig:
    sensor_zip: Path
    channel_zip: Path
    condition: str = "sunny"
    town: str = DEFAULT_TOWN
    scenario: str = DEFAULT_SCENARIO
    output_root: Path = Path("dataset")
    seq_len: int = 8
    pred_len: int = 3
    num_beams: int = 64
    tx_antennas: int = 64
    rx_antennas: int = 1
    group_size: int = 8
    split_seed: int = 42
    train_ratio: float = 0.8
    split_tag: str = ""
    split_strategy: str = GROUP_SAFE_TIME_BLOCK
    block_size_frames: int | None = None
    guard_band_frames: int | None = None
    enabled_modalities: tuple[str, ...] = ("camera0", "lidar", "gps", "channel")
    channel_scenario: str | None = None
    channel_scenario_aliases: dict[str, str] = field(default_factory=dict)
    radio_semantic: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "MMWPreparationConfig":
        cfg = dict(payload.get("mmw", payload.get("preprocessing", payload)))
        return cls(
            sensor_zip=Path(str(cfg.get("sensor_zip", ""))).expanduser(),
            channel_zip=Path(str(cfg.get("channel_zip", ""))).expanduser(),
            condition=str(cfg.get("condition", "sunny")),
            town=str(cfg.get("town", DEFAULT_TOWN)),
            scenario=str(cfg.get("scenario", DEFAULT_SCENARIO)),
            output_root=Path(str(cfg.get("output_root", "dataset"))).expanduser(),
            seq_len=int(cfg.get("seq_len", 8)),
            pred_len=int(cfg.get("pred_len", cfg.get("num_pred", 3))),
            num_beams=int(cfg.get("num_beams", 64)),
            tx_antennas=int(cfg.get("tx_antennas", cfg.get("num_tx_antennas", 64))),
            rx_antennas=int(cfg.get("rx_antennas", cfg.get("num_rx_antennas", 1))),
            group_size=int(cfg.get("group_size", 8)),
            split_seed=int(cfg.get("split_seed", 42)),
            train_ratio=float(cfg.get("train_ratio", 0.8)),
            split_tag=str(cfg.get("split_tag", cfg.get("sequence_tag", "")) or ""),
            split_strategy=str(cfg.get("split_strategy", GROUP_SAFE_TIME_BLOCK) or GROUP_SAFE_TIME_BLOCK),
            block_size_frames=(
                int(cfg["block_size_frames"])
                if cfg.get("block_size_frames") is not None
                else None
            ),
            guard_band_frames=(
                int(cfg["guard_band_frames"])
                if cfg.get("guard_band_frames") is not None
                else None
            ),
            enabled_modalities=tuple(str(item) for item in cfg.get("enabled_modalities", ("camera0", "lidar", "gps", "channel"))),
            channel_scenario=str(cfg["channel_scenario"]) if cfg.get("channel_scenario") else None,
            channel_scenario_aliases={
                str(key): str(value)
                for key, value in (cfg.get("channel_scenario_aliases") or {}).items()
            },
            radio_semantic=dict(cfg.get("radio_semantic") or {}),
        )

    @property
    def condition_root(self) -> Path:
        return self.output_root / "MMW" / self.condition

    @property
    def sensor_root(self) -> Path:
        return self.condition_root / "Sensor_Data"

    @property
    def channel_root(self) -> Path:
        return self.condition_root / "Channel_Data"

    @property
    def prepared_root(self) -> Path:
        return self.condition_root / "Prepared" / self.scenario

    @property
    def resolved_channel_scenario(self) -> str:
        if self.channel_scenario:
            return self.channel_scenario
        alias = self.channel_scenario_aliases.get(self.scenario)
        if alias:
            return alias
        from kd_sensing.data.mmw.preparation_index import default_channel_scenario
        return default_channel_scenario(self.scenario)


def load_preparation_config(config_path: str | Path, overrides: list[str] | None = None) -> MMWPreparationConfig:
    payload = safe_load_yaml(Path(config_path).read_text(encoding="utf-8")) or {}
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return MMWPreparationConfig.from_mapping(payload)

__all__ = [
    'ALGORITHM_VERSION',
    'DEFAULT_SCENARIO',
    'DEFAULT_TOWN',
    'GROUP_SAFE_TIME_BLOCK',
    'MMWPreparationConfig',
    'MMW_SPLIT_PROTOCOL_VERSION',
    'SUPPORTED_SEQUENCE_SPLIT_STRATEGIES',
    'load_preparation_config'
]
