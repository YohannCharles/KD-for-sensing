"""Virtual canonical config path routing and retired-name guards."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kd_sensing.utils.paths import project_root

REMOVED_FUSION_CONFIG_STEMS = {
    "no_kd": "image_radar_lightweight.yaml",
}
RETIRED_FUSION_KD_MODES = ("logits_kd", "rkd", "teacher_no_kd", "student_no_kd")
_RETIRED_FUSION_KD_SUFFIXES = tuple((f"_{mode}", mode) for mode in RETIRED_FUSION_KD_MODES)


@dataclass(frozen=True)
class VirtualConfigBuilders:
    fusion: Callable[[str], dict[str, Any]]
    snapshot_single: Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class VirtualConfigRoutes:
    snapshot_mode: str
    single_modalities: Sequence[str]
    removed_fusion_config_stems: Mapping[str, str] = field(default_factory=lambda: dict(REMOVED_FUSION_CONFIG_STEMS))


def build_virtual_config_for_path(
    config_path: Path,
    *,
    builders: VirtualConfigBuilders,
    routes: VirtualConfigRoutes,
) -> dict[str, Any] | None:
    """Build a virtual canonical config override for a missing config path."""

    if is_fusion_config_path(config_path):
        replacement = routes.removed_fusion_config_stems.get(config_path.stem)
        if replacement is not None:
            raise ValueError(
                f"Removed fusion config alias '{config_path.name}'. "
                f"Use 'configs/fusion/{replacement}' instead."
            )
        return builders.fusion(config_path.stem)
    single = parse_single_snapshot_config_path(
        config_path,
        snapshot_mode=routes.snapshot_mode,
        single_modalities=routes.single_modalities,
    )
    if single is not None:
        return builders.snapshot_single(single)
    return None


def ensure_not_retired_fusion_kd_config(stem: str) -> None:
    retired_kd_mode = retired_fusion_kd_mode(stem)
    if retired_kd_mode is not None:
        raise ValueError(
            f"KD support has been removed for legacy fusion config '{stem}.yaml' "
            f"({retired_kd_mode}). Use '<slug>_strong.yaml' or '<slug>_lightweight.yaml'."
        )


def retired_fusion_kd_mode(stem: str) -> str | None:
    if stem in RETIRED_FUSION_KD_MODES:
        return stem
    for suffix, mode in _RETIRED_FUSION_KD_SUFFIXES:
        if stem.endswith(suffix):
            return mode
    return None


def parse_single_snapshot_config_path(
    path: Path,
    *,
    snapshot_mode: str,
    single_modalities: Sequence[str],
) -> str | None:
    if path.suffix not in {".yaml", ".yml"} or path.stem != snapshot_mode:
        return None
    parts = config_path_parts(path)
    if len(parts) != 3 or parts[0] != "configs":
        return None
    modality = parts[1]
    if modality == "fusion":
        return None
    if modality not in set(single_modalities):
        return None
    return modality


def is_fusion_config_path(path: Path) -> bool:
    if path.suffix not in {".yaml", ".yml"}:
        return False
    parts = config_path_parts(path)
    return len(parts) == 3 and parts[:2] == ("configs", "fusion")


def config_path_parts(path: Path) -> tuple[str, ...]:
    try:
        relative = path.resolve().relative_to(project_root())
        return tuple(relative.parts)
    except ValueError:
        parts = path.parts
        if len(parts) >= 3 and parts[-3] == "configs":
            return tuple(parts[-3:])
        return tuple(parts)
