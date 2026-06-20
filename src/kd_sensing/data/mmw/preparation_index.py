from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kd_sensing.data.mmw.preparation_config import DEFAULT_SCENARIO, DEFAULT_TOWN, MMWPreparationConfig



@dataclass
class SensorFrame:
    agent: str
    frame_id: str
    yaml_path: Path | None = None
    lidar_path: Path | None = None
    cameras: dict[str, Path] = field(default_factory=dict)
    depth_cameras: dict[str, Path] = field(default_factory=dict)
    radar_path: Path | None = None
    rsu: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedFrame:
    condition: str = "sunny"
    town: str = DEFAULT_TOWN
    sensor_scenario: str = DEFAULT_SCENARIO
    channel_scenario: str = "Town10_skybridge"
    agent: str = ""
    channel_agent: str = ""
    frame_id: str = ""
    sample_id: str = ""
    camera0: str = ""
    cameras: dict[str, str] = field(default_factory=dict)
    depth_cameras: dict[str, str] = field(default_factory=dict)
    lidar: str = ""
    gps: str = ""
    radar: str = ""
    channel_path: str = ""
    beam_power_path: str = ""
    beam_label: int = 0
    coarse_sector: int = 0
    radio_semantic_label: int | None = None
    radio_semantic_available: bool = False
    radio_semantic_unavailable_reason: str = ""
    radio_semantic_metadata: dict[str, Any] = field(default_factory=dict)
    modality_availability: dict[str, Any] = field(default_factory=dict)
    relative_geometry: dict[str, Any] = field(default_factory=dict)
    proxy_features: dict[str, Any] = field(default_factory=dict)
    channel_fields: dict[str, Any] = field(default_factory=dict)
    rsu: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelFile:
    path: Path
    agent: str | None
    frame_id: str
    scenario: str


def index_sensor_frames(sensor_root: str | Path, *, town: str, scenario: str) -> dict[str, dict[str, SensorFrame]]:
    root = Path(sensor_root)
    scenario_root = _find_scenario_root(root, town=town, scenario=scenario)
    frames: dict[str, dict[str, SensorFrame]] = defaultdict(dict)
    rsu_by_frame: dict[str, dict[str, Any]] = defaultdict(lambda: {"agents": {}})
    for path in scenario_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(scenario_root).parts
        if not rel_parts:
            continue
        agent = rel_parts[0]
        frame_id = _frame_id_from_path(path)
        if frame_id is None:
            continue
        kind = _sensor_kind(path)
        if _is_rsu_agent(agent):
            entry = rsu_by_frame[frame_id]["agents"].setdefault(agent, {})
            if kind:
                entry[kind] = path
            continue
        if not _is_cav_agent(agent):
            continue
        frame = frames[agent].setdefault(frame_id, SensorFrame(agent=agent, frame_id=frame_id))
        if kind == "yaml":
            frame.yaml_path = path
        elif kind == "lidar":
            frame.lidar_path = path
        elif kind and kind.startswith("camera"):
            frame.cameras[kind] = path
        elif kind and kind.startswith("depth"):
            frame.depth_cameras[kind] = path
        elif kind == "radar":
            frame.radar_path = path
    for agent_frames in frames.values():
        for frame in agent_frames.values():
            if frame.frame_id in rsu_by_frame:
                frame.rsu = rsu_by_frame[frame.frame_id]
    return {agent: dict(agent_frames) for agent, agent_frames in frames.items()}


def index_channel_files(
    channel_root: str | Path,
    *,
    town: str,
    scenario: str,
    channel_scenario: str | None = None,
) -> dict[tuple[str | None, str], ChannelFile]:
    root = Path(channel_root)
    search_roots = _channel_search_roots(root, town=town, scenario=scenario, channel_scenario=channel_scenario)
    index: dict[tuple[str | None, str], ChannelFile] = {}
    for path in _unique_channel_paths(search_roots):
        if path.suffix.lower() not in {".npy", ".npz"}:
            continue
        frame_id = _frame_id_from_path(path)
        if frame_id is None:
            continue
        agent = _agent_from_path_parts(path.parts)
        scenario_name = _channel_scenario_from_path(path.parts, town=town) or str(channel_scenario or scenario)
        item = ChannelFile(path=path, agent=agent, frame_id=frame_id, scenario=scenario_name)
        index[(agent, frame_id)] = item
        if agent is None:
            index.setdefault((None, frame_id), item)
    return index


def _find_scenario_root(root: Path, *, town: str, scenario: str) -> Path:
    direct = root / town / scenario
    if direct.exists():
        return direct
    matches = [path for path in root.rglob(scenario) if path.is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find MMW scenario root '{town}/{scenario}' under {root.resolve()}.")


def default_channel_scenario(sensor_scenario: str) -> str:
    parts = str(sensor_scenario).split("_")
    if len(parts) >= 2 and parts[-1].startswith("seed") and parts[-1][4:].isdigit():
        return "_".join(parts[:-1])
    return str(sensor_scenario)


def sample_id_for(condition: str, town: str, scenario: str, agent: str, frame_id: str) -> str:
    return f"{condition}:{town}:{scenario}:{agent}:{frame_id}"


def _frame_id_from_path(path: Path) -> str | None:
    for token in [path.stem, *reversed(path.parts)]:
        for part in str(token).replace("-", "_").split("_"):
            if part.isdigit() and len(part) == 6:
                return part
    return None


def _agent_from_scenario_path(path: Path, scenario_root: Path) -> str | None:
    try:
        rel = path.relative_to(scenario_root)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def _agent_from_path_parts(parts: tuple[str, ...]) -> str | None:
    for part in reversed(parts):
        if _is_cav_agent(part):
            return part
    return None


def _channel_scenario_from_path(parts: tuple[str, ...], *, town: str) -> str | None:
    for index, part in enumerate(parts):
        if part == town and index + 1 < len(parts):
            return parts[index + 1]
    for part in parts:
        if part.startswith(f"{town}_"):
            return part
    return None


def _channel_search_roots(
    root: Path,
    *,
    town: str,
    scenario: str,
    channel_scenario: str | None,
) -> list[Path]:
    candidates = []
    for name in (channel_scenario, default_channel_scenario(scenario), scenario):
        if not name:
            continue
        direct = root / town / name
        if direct.exists():
            candidates.append(direct)
        candidates.extend(path for path in root.rglob(str(name)) if path.is_dir())
    if not candidates:
        fallback = root / town if (root / town).exists() else root
        candidates.append(fallback)
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _unique_channel_paths(search_roots: list[Path]) -> list[Path]:
    seen = set()
    paths = []
    for root in search_roots:
        for path in root.rglob("*_paths.*"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def _sensor_kind(path: Path) -> str | None:
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".pcd":
        return "lidar"
    if suffix in {".png", ".jpg", ".jpeg"}:
        for camera_idx in range(4):
            if f"camera{camera_idx}" in stem:
                return f"camera{camera_idx}"
        for depth_idx in range(4):
            if f"depth_camera{depth_idx}" in stem or f"depth{depth_idx}" in stem:
                return f"depth_camera{depth_idx}"
        if "depth" in stem:
            return "depth_camera"
        if "camera" in stem:
            return "camera"
    if suffix == ".json":
        return "radar"
    return None


def _is_rsu_agent(agent: str) -> bool:
    key = agent.lower()
    return key.startswith("rsu") or "infrastructure" in key or "roadside" in key


def _is_cav_agent(agent: str) -> bool:
    return agent.lower().startswith("cav")


def _missing_required_modalities(frame: SensorFrame, *, enabled: tuple[str, ...]) -> list[str]:
    missing = []
    if "camera0" in enabled and "camera0" not in frame.cameras:
        missing.append("missing_camera0")
    if "lidar" in enabled and frame.lidar_path is None:
        missing.append("missing_lidar")
    if "gps" in enabled and frame.yaml_path is None:
        missing.append("missing_metadata")
    return missing

__all__ = [
    'SensorFrame',
    'PreparedFrame',
    'ChannelFile',
    'index_sensor_frames',
    'index_channel_files',
    'default_channel_scenario',
    'sample_id_for'
]
