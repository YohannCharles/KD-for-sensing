from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.evaluation.subset_specs import (  # noqa: E402
    generic_modality_subset_specs,
    resolve_named_modality_subset,
    subset_mask,
    subset_metadata,
)


MODEL_MODALITIES = ("image", "gps", "lidar", "mmwave")


def test_generic_modality_subsets_are_named_and_ordered():
    specs = generic_modality_subset_specs(MODEL_MODALITIES)

    assert list(specs) == [
        "all",
        "strong_only",
        "gps_mmwave",
        "weak_only",
        "image",
        "gps",
        "lidar",
        "mmwave",
    ]
    assert specs["all"].modalities == MODEL_MODALITIES
    assert specs["strong_only"].modalities == ("gps", "mmwave")
    assert specs["weak_only"].modalities == ("image", "lidar")


def test_generic_subset_metadata_includes_masks_and_combinations():
    spec = resolve_named_modality_subset("image_lidar", MODEL_MODALITIES)
    assert spec is not None
    assert spec.modalities == ("image", "lidar")
    assert spec.mask_for(MODEL_MODALITIES) == (True, False, True, False)
    assert subset_mask("weak_only", MODEL_MODALITIES) == (True, False, True, False)

    metadata = subset_metadata(MODEL_MODALITIES)
    assert len(metadata) == 8
    assert metadata[0]["name"] == "all"
    assert metadata[0]["mask"] == [True, True, True, True]
    assert resolve_named_modality_subset("strong_plus_lidar", MODEL_MODALITIES) is None
