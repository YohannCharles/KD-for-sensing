from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities


@dataclass(frozen=True)
class ModalitySubsetSpec:
    name: str
    modalities: tuple[str, ...]

    def mask_for(self, model_modalities: Iterable[str]) -> tuple[bool, ...]:
        selected = set(self.modalities)
        normalized = normalize_modalities(tuple(model_modalities), context=f"{self.name} model modalities")
        missing = sorted(selected.difference(normalized))
        if missing:
            raise ValueError(
                f"Subset '{self.name}' requires modalities {missing}, "
                f"but model modalities are {list(normalized)}."
            )
        return tuple(name in selected for name in normalized)

    def to_metadata(self, model_modalities: Iterable[str] | None = None) -> dict:
        metadata = {
            "name": self.name,
            "modalities": list(self.modalities),
        }
        if model_modalities is not None:
            metadata["mask"] = list(self.mask_for(model_modalities))
        return metadata


SCENE32_CONDITIONAL_UTILITY_SUBSETS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    (
        ("all", normalize_modalities(MODALITY_ORDER, context="conditional audit subset all")),
        ("strong_only", normalize_modalities(("gps", "mmwave"), context="conditional audit subset strong_only")),
        (
            "strong_plus_image",
            normalize_modalities(("gps", "mmwave", "image"), context="conditional audit subset strong_plus_image"),
        ),
        (
            "strong_plus_radar",
            normalize_modalities(("gps", "mmwave", "radar"), context="conditional audit subset strong_plus_radar"),
        ),
        (
            "strong_plus_lidar",
            normalize_modalities(("gps", "mmwave", "lidar"), context="conditional audit subset strong_plus_lidar"),
        ),
        (
            "single_best_mmwave",
            normalize_modalities(("mmwave",), context="conditional audit subset single_best_mmwave"),
        ),
        ("weak_only", normalize_modalities(("image", "radar", "lidar"), context="conditional audit subset weak_only")),
    )
)

CONDITIONAL_UTILITY_SUBSET_NAMES = tuple(SCENE32_CONDITIONAL_UTILITY_SUBSETS.keys())


def scene32_conditional_utility_subset_specs() -> "OrderedDict[str, ModalitySubsetSpec]":
    return OrderedDict(
        (name, ModalitySubsetSpec(name=name, modalities=modalities))
        for name, modalities in SCENE32_CONDITIONAL_UTILITY_SUBSETS.items()
    )


def resolve_conditional_utility_subset(
    name: str,
    model_modalities: Iterable[str],
) -> ModalitySubsetSpec | None:
    raw = SCENE32_CONDITIONAL_UTILITY_SUBSETS.get(str(name))
    if raw is None:
        return None
    normalized_model_modalities = normalize_modalities(tuple(model_modalities), context="conditional audit model modalities")
    if str(name) == "all":
        return ModalitySubsetSpec(name="all", modalities=normalized_model_modalities)
    if not set(raw).issubset(set(normalized_model_modalities)):
        return None
    return ModalitySubsetSpec(name=str(name), modalities=raw)


def subset_mask(name: str, model_modalities: Iterable[str]) -> tuple[bool, ...]:
    spec = resolve_conditional_utility_subset(name, model_modalities)
    if spec is None:
        raise KeyError(f"Unknown or unavailable conditional utility subset '{name}'.")
    return spec.mask_for(model_modalities)


def subset_metadata(model_modalities: Iterable[str]) -> list[dict]:
    specs = scene32_conditional_utility_subset_specs()
    metadata = []
    for name in specs:
        spec = resolve_conditional_utility_subset(name, model_modalities)
        if spec is not None:
            metadata.append(spec.to_metadata(model_modalities))
    return metadata


__all__ = [
    "CONDITIONAL_UTILITY_SUBSET_NAMES",
    "ModalitySubsetSpec",
    "SCENE32_CONDITIONAL_UTILITY_SUBSETS",
    "resolve_conditional_utility_subset",
    "scene32_conditional_utility_subset_specs",
    "subset_mask",
    "subset_metadata",
]
