from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from kd_sensing.modalities import normalize_modalities


GENERIC_STRONG_MODALITIES = ("gps", "mmwave")
GENERIC_WEAK_MODALITIES = ("image", "radar", "lidar")


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


def generic_modality_subset_specs(model_modalities: Iterable[str]) -> "OrderedDict[str, ModalitySubsetSpec]":
    normalized = normalize_modalities(tuple(model_modalities), context="generic modality subset model modalities")
    specs: "OrderedDict[str, ModalitySubsetSpec]" = OrderedDict()
    specs["all"] = ModalitySubsetSpec(name="all", modalities=normalized)
    strong = _available_modalities(GENERIC_STRONG_MODALITIES, normalized)
    if strong:
        specs["strong_only"] = ModalitySubsetSpec(name="strong_only", modalities=strong)
        specs["gps_mmwave"] = ModalitySubsetSpec(name="gps_mmwave", modalities=strong)
    weak = _available_modalities(GENERIC_WEAK_MODALITIES, normalized)
    if weak:
        specs["weak_only"] = ModalitySubsetSpec(name="weak_only", modalities=weak)
    for modality in normalized:
        specs[modality] = ModalitySubsetSpec(name=modality, modalities=(modality,))
    return specs


def resolve_named_modality_subset(name: str, model_modalities: Iterable[str]) -> ModalitySubsetSpec | None:
    requested = str(name)
    normalized = normalize_modalities(tuple(model_modalities), context="generic modality subset model modalities")
    specs = generic_modality_subset_specs(normalized)
    if requested in specs:
        return specs[requested]
    parts = tuple(part for part in requested.split("_") if part)
    if parts and all(part in normalized for part in parts):
        return ModalitySubsetSpec(
            name=requested,
            modalities=normalize_modalities(parts, context=f"generic modality subset {requested}"),
        )
    return None


def subset_mask(name: str, model_modalities: Iterable[str]) -> tuple[bool, ...]:
    spec = resolve_named_modality_subset(name, model_modalities)
    if spec is None:
        raise KeyError(f"Unknown or unavailable modality subset '{name}'.")
    return spec.mask_for(model_modalities)


def subset_metadata(model_modalities: Iterable[str]) -> list[dict]:
    return [spec.to_metadata(model_modalities) for spec in generic_modality_subset_specs(model_modalities).values()]


def _available_modalities(candidates: Iterable[str], model_modalities: Iterable[str]) -> tuple[str, ...]:
    available = set(model_modalities)
    return tuple(name for name in normalize_modalities(tuple(candidates), context="generic modality subset candidates") if name in available)


__all__ = [
    "GENERIC_STRONG_MODALITIES",
    "GENERIC_WEAK_MODALITIES",
    "ModalitySubsetSpec",
    "generic_modality_subset_specs",
    "resolve_named_modality_subset",
    "subset_mask",
    "subset_metadata",
]
