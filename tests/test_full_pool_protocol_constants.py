"""Pin the canonical Full-pool split constants.

The local Full-pool experiment tools (`run_full_pool_bt_scl`,
`run_full_pool_candidate12`, `run_full_pool_btma_ablation`) used to restate
these window counts as literals in their own audit assertions.  They now read
them from `full_pool_protocol`, which makes that module a single point of
failure: a typo there would propagate into every workflow's audit output at
once.

These expectations are therefore written as independent literals on purpose.
Do NOT rewrite them to import the constants they guard -- that would make the
assertions tautological.  Changing the protocol means changing both this file
and the protocol fingerprint, deliberately and together.
"""

from __future__ import annotations

from kd_sensing.data.mmw.full_pool_protocol import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_DOMAIN_COUNT,
    EXPECTED_HISTORICAL_EXCLUSION_COUNT,
    FULL_POOL_DEVELOPMENT_WINDOWS,
    FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
    FULL_POOL_PROTOCOL_ID,
    FULL_POOL_RAW_TRAIN_COUNT,
    FULL_POOL_RESOURCE_INTERSECTION_NAMES,
    FULL_POOL_SPLIT_EXPECTATIONS,
)


def test_split_expectations_match_the_canonical_protocol() -> None:
    assert FULL_POOL_SPLIT_EXPECTATIONS == {
        "candidate_window_count": 46_860,
        "boundary_crossing_excluded_count": 240,
        "historical_removed_from_train_count": 402,
        "historical_protected_count": 588,
        "train_sample_count": 37_038,
        "validation_sample_count": 9_180,
    }
    assert FULL_POOL_PROTOCOL_ID == "mmw_full_pool_development_v1"
    assert EXPECTED_CANDIDATE_COUNT == 46_860
    assert EXPECTED_DOMAIN_COUNT == 15
    assert EXPECTED_HISTORICAL_EXCLUSION_COUNT == 588


def test_derived_counts_are_arithmetically_consistent() -> None:
    expectations = FULL_POOL_SPLIT_EXPECTATIONS
    assert FULL_POOL_DEVELOPMENT_WINDOWS == 46_218
    assert FULL_POOL_RAW_TRAIN_COUNT == 37_440
    assert FULL_POOL_HISTORICAL_VALIDATION_RETAINED == 186
    assert FULL_POOL_DEVELOPMENT_WINDOWS == (
        expectations["train_sample_count"] + expectations["validation_sample_count"]
    )
    assert FULL_POOL_RAW_TRAIN_COUNT == (
        expectations["train_sample_count"] + expectations["historical_removed_from_train_count"]
    )
    assert FULL_POOL_HISTORICAL_VALIDATION_RETAINED == (
        expectations["historical_protected_count"]
        - expectations["historical_removed_from_train_count"]
    )
    assert FULL_POOL_DEVELOPMENT_WINDOWS < expectations["candidate_window_count"]


def test_audit_strings_render_without_digit_separators() -> None:
    """The tools interpolate these into audit text that downstream parsers read."""
    expectations = FULL_POOL_SPLIT_EXPECTATIONS
    train = expectations["train_sample_count"]
    validation = expectations["validation_sample_count"]
    assert f"protocol_counts={train},{validation},{FULL_POOL_DEVELOPMENT_WINDOWS}" == (
        "protocol_counts=37038,9180,46218"
    )
    assert f"{train:,}/{validation:,}" == "37,038/9,180"


def test_resource_intersection_names_cover_every_identity_family() -> None:
    assert FULL_POOL_RESOURCE_INTERSECTION_NAMES == (
        "sample_id",
        "target_sample_id",
        "full_csv_row",
        "all_frame_dependency",
        "camera_resource",
        "lidar_resource",
        "radar_resource",
        "ue_gps_resource",
        "bs_gps_resource",
        "channel_resource",
    )
    assert len(set(FULL_POOL_RESOURCE_INTERSECTION_NAMES)) == len(
        FULL_POOL_RESOURCE_INTERSECTION_NAMES
    )
