from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.evaluation.subset_specs import (  # noqa: E402
    CONDITIONAL_UTILITY_SUBSETS,
    CONDITIONAL_UTILITY_SUBSET_NAMES,
    resolve_conditional_utility_subset,
    subset_metadata,
)
from kd_sensing.modalities import MODALITY_ORDER  # noqa: E402


def test_conditional_utility_subsets_are_named_and_ordered():
    assert CONDITIONAL_UTILITY_SUBSET_NAMES == (
        "all",
        "strong_only",
        "strong_plus_image",
        "strong_plus_radar",
        "strong_plus_lidar",
        "single_best_mmwave",
        "weak_only",
    )
    assert list(CONDITIONAL_UTILITY_SUBSETS) == list(CONDITIONAL_UTILITY_SUBSET_NAMES)
    for modalities in CONDITIONAL_UTILITY_SUBSETS.values():
        indices = [MODALITY_ORDER.index(name) for name in modalities]
        assert indices == sorted(indices)


def test_conditional_subset_metadata_includes_masks():
    spec = resolve_conditional_utility_subset("strong_plus_lidar", MODALITY_ORDER)
    assert spec is not None
    assert spec.modalities == ("gps", "lidar", "mmwave")
    assert spec.mask_for(MODALITY_ORDER) == (False, False, True, True, True)

    metadata = subset_metadata(MODALITY_ORDER)
    assert len(metadata) == 7
    assert metadata[0]["name"] == "all"
    assert metadata[0]["mask"] == [True, True, True, True, True]
