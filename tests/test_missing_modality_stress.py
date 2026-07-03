import pytest

from kd_sensing.diagnostics.missing_modality_stress import (
    baseline_stress_comparability_metadata,
    canonical_missing_modality_conditions,
    normalize_missing_modality_stress_manifest,
)


def test_stress_manifest_normalizes_smoke_taxonomy_and_claim_boundary():
    manifest = normalize_missing_modality_stress_manifest(
        {
            "suite": "missing_modality_stress",
            "tier": "smoke",
            "modalities": ["image", "radar", "gps", "lidar", "mmwave"],
            "model_groups": [
                {
                    "id": "amber_lite",
                    "config_path": "configs/fusion/amber_lite_missing_modality.yaml",
                    "baseline_scope": "local experimental baseline",
                }
            ],
        }
    )

    ids = {condition["id"] for condition in manifest["conditions"]}
    assert {"full", "missing_image", "missing_gps", "gps_only", "non_gps_only", "avg_missing"} <= ids
    assert {"unavailable_lidar", "unavailable_mmwave"} <= ids
    assert any(condition["condition_type"] == "random_missing" for condition in manifest["conditions"])
    unavailable = next(condition for condition in manifest["conditions"] if condition["id"] == "unavailable_mmwave")
    assert unavailable["difficulty_profile"]["operators"][0]["type"] == "modality_unavailable"
    assert manifest["claim_status"] == "mock/smoke"
    assert manifest["output_dir"].startswith("outputs/analysis/missing_modality_stress")
    assert manifest["model_groups"][0]["strict_comparability"]["status"] == "not_comparable"
    assert any(warning["code"] == "strict_comparability_missing" for warning in manifest["warnings"])


def test_stress_manifest_rejects_unknown_condition_with_available_list():
    with pytest.raises(ValueError, match="Available conditions"):
        normalize_missing_modality_stress_manifest(
            {
                "suite": "missing_modality_stress",
                "conditions": ["missing_thermal_camera"],
            }
        )


def test_baseline_comparability_metadata_marks_strict_and_missing_fields():
    strict = baseline_stress_comparability_metadata(
        model_group="U-MaskBeamJEPA",
        config_path="configs/fusion/u_mask_beam_jepa_smoke.yaml",
        weights_path="outputs/local/best.pth",
        checkpoint_provenance="local checkpoint",
        modalities=["image", "radar", "gps", "lidar"],
        split="test",
        sample_count=16,
        label_space="beam64",
        metric_profile="topk_dba",
        target_source="current",
        seed=17,
        difficulty_digest="digest",
    )
    missing = baseline_stress_comparability_metadata(
        model_group="RMBP-MM",
        config_path="configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml",
    )

    assert strict["strict_comparability"]["status"] == "strict"
    assert strict["eligible_for_strict_claim"] is True
    assert missing["strict_comparability"]["status"] == "not_comparable"
    assert "weights_path" in missing["strict_comparability"]["missing_fields"]


def test_canonical_conditions_cover_mmwave_without_reading_dataset():
    conditions = canonical_missing_modality_conditions(["image", "gps", "mmwave"], severities=[0.25])

    ids = {condition["id"] for condition in conditions}
    assert {"missing_mmwave", "mmwave_only", "unavailable_mmwave", "random_missing_p0p25"} <= ids
