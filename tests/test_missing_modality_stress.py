import csv
import importlib.util
import py_compile
from pathlib import Path

import pytest

from kd_sensing.diagnostics.missing_modality_stress import (
    baseline_stress_comparability_metadata,
    canonical_missing_modality_conditions,
    normalize_missing_modality_stress_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE31_34_FINAL_SCRIPTS = (
    "summarize_scenes31_34_main.py",
    "plot_missing_count_degradation.py",
    "profile_scenes31_34_methods.py",
    "export_scenes31_34_main_paper_tables.py",
    "write_scenes31_34_main_conclusion.py",
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


def test_scene31_34_final_scripts_compile_and_emit_pending_caveats(tmp_path: Path):
    for script in SCENE31_34_FINAL_SCRIPTS:
        py_compile.compile(str(ROOT / "scripts" / script), doraise=True)

    summary_mod = _load_script("summarize_scenes31_34_main", ROOT / "scripts" / "summarize_scenes31_34_main.py")
    export_mod = _load_script("export_scenes31_34_main_paper_tables", ROOT / "scripts" / "export_scenes31_34_main_paper_tables.py")
    conclusion_mod = _load_script("write_scenes31_34_main_conclusion", ROOT / "scripts" / "write_scenes31_34_main_conclusion.py")

    summary_root = tmp_path / "summary"
    paper_root = tmp_path / "paper_tables"
    conclusion_path = tmp_path / "final_main_conclusion.txt"
    summary_mod.summarize(tmp_path / "empty_root", summary_root, [])
    export_mod.main(["--summary-root", str(summary_root), "--profile-root", str(tmp_path / "profile"), "--out", str(paper_root)])
    conclusion_mod.main(
        [
            "--summary-root",
            str(summary_root),
            "--paper-table-root",
            str(paper_root),
            "--profile-root",
            str(tmp_path / "profile"),
            "--out",
            str(conclusion_path),
        ]
    )

    checklist = list(csv.DictReader((summary_root / "final_evidence_checklist.csv").open("r", encoding="utf-8", newline="")))
    assert any(row["item"] == "ordinary classifier baseline" and row["status"] == "pending" for row in checklist)
    assert "pending" in (paper_root / "table_scenes31_34_classifier_baseline.md").read_text(encoding="utf-8")
    conclusion = conclusion_path.read_text(encoding="utf-8")
    assert "Final evidence is not yet complete" in conclusion


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
