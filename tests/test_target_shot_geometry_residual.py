from __future__ import annotations

import json
from pathlib import Path

import pytest

from kd_sensing.data.dataset_runtime import RuntimeDataset, SampleIndex, SampleRow
from kd_sensing.data.geometry_residual import (
    GeometryResidualTargetProvider,
    beam_to_residual,
    circular_beam_distance,
    make_residual_class,
    residual_class_to_delta,
    residual_to_beam,
)
from kd_sensing.data.target_shot_runtime import (
    TargetShotSupervisionGuardError,
    assert_target_supervision_allowed,
    target_shot_runtime_metadata,
)
from kd_sensing.data.target_shot_splits import (
    TargetShotSplitConfig,
    build_domain_key,
    build_target_shot_split,
    validate_target_shot_artifact,
    write_target_shot_artifact,
)


def test_target_shot_split_reproducible_five_percent_and_artifact_mismatch(tmp_path: Path):
    rows = _target_shot_rows()
    cfg = TargetShotSplitConfig(
        domain_type="scenario_weather",
        source_domains=("src:sunny",),
        target_domains=("target:rain",),
        target_label_fraction=0.05,
        target_label_selection="stratified_by_beam",
        seed=123,
    )

    first = build_target_shot_split(rows, cfg, dataset_type="mmw")
    second = build_target_shot_split(rows, cfg, dataset_type="mmw")

    assert first["splits"]["target_labeled"]["count"] == 5
    assert first["splits"]["target_labeled"]["sample_ids"] == second["splits"]["target_labeled"]["sample_ids"]
    split_sets = [set(payload["sample_ids"]) for payload in first["splits"].values()]
    for idx, left in enumerate(split_sets):
        for right in split_sets[idx + 1 :]:
            assert left.isdisjoint(right)
    assert first["sampling_manifest"]["buckets"]
    assert first["leakage_diagnostics"]["sample_id_overlap_count"] == 0

    paths = write_target_shot_artifact(first, tmp_path / "split.json")
    assert Path(paths["json"]).exists()
    assert Path(paths["npz"]).exists()
    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    validate_target_shot_artifact(loaded, rows, cfg)
    changed = TargetShotSplitConfig(
        domain_type="scenario_weather",
        source_domains=("src:sunny",),
        target_domains=("target:rain",),
        target_label_fraction=0.10,
        target_label_selection="stratified_by_beam",
        seed=123,
    )
    with pytest.raises(ValueError, match="target_label_fraction.*Regenerate"):
        validate_target_shot_artifact(loaded, rows, changed)


def test_domain_key_missing_field_reports_dataset_type():
    with pytest.raises(ValueError, match="dataset_type=mmw.*missing field weather/condition"):
        build_domain_key({"sample_id": "x", "scenario": "town"}, domain_type="scenario_weather", dataset_type="mmw")


def test_geometry_residual_wraparound_inverse_and_clipped_class():
    assert circular_beam_distance(0, 63, num_beams=64) == 1
    residual = beam_to_residual(beam_abs=0, beam_geo=63, num_beams=64, convention="signed_circular")
    assert residual == 1
    assert residual_to_beam(residual, beam_geo=63, num_beams=64, convention="signed_circular") == 0
    full = beam_to_residual(beam_abs=0, beam_geo=63, num_beams=64, convention="full_circular")
    assert full == 1

    inside = make_residual_class(-3, max_residual=8, overflow_strategy="boundary")
    assert inside.class_id == 5
    assert residual_class_to_delta(inside.class_id, max_residual=8, overflow_strategy="boundary") == -3
    overflow = make_residual_class(12, max_residual=8, overflow_strategy="boundary")
    assert overflow.overflow is True
    assert overflow.class_id == 16
    ignored = make_residual_class(12, max_residual=8, overflow_strategy="ignore", ignore_index=-100)
    assert ignored.class_id == -100
    assert residual_class_to_delta(-100, max_residual=8, overflow_strategy="ignore", ignore_index=-100) is None


def test_runtime_geometry_residual_keys_absolute_compatibility_and_unlabeled_guard():
    rows = (
        SampleRow(
            sample_id="target-0",
            split="target_labeled",
            dataset_type="mmw",
            family="MMW",
            target_ref={"beam_abs": 3},
            metadata={"relative_geometry": {"available": True, "relative_azimuth": 90.0}},
        ),
    )
    dataset = RuntimeDataset(
        sample_index=SampleIndex.from_rows(rows, storage_kind="unit", metadata={"target_shot": {"target_labeled_count": 1}}),
        modality_adapters=(),
        target_provider=GeometryResidualTargetProvider(num_beams=8, max_residual=2, num_geo_sectors=4),
        dataset_type="mmw",
        descriptor={"family": "MMW", "storage_kind": "unit"},
        enabled_modalities=(),
        input_profiles={},
        return_metadata=True,
    )

    sample = dataset[0]
    assert {"beam_abs", "beam_geo", "beam_residual", "residual_class", "geo_angle", "geo_sector"} <= set(sample)
    assert dataset.runtime_metadata()["target_schema"] == "geometry_residual"
    assert dataset.runtime_metadata()["target_shot"]["target_labeled_count"] == 1

    absolute_dataset = RuntimeDataset(
        sample_index=SampleIndex.from_rows(rows, storage_kind="unit"),
        modality_adapters=(),
        target_provider=_AbsoluteProvider(),
        dataset_type="mmw",
        descriptor={"family": "MMW", "storage_kind": "unit"},
        enabled_modalities=(),
        input_profiles={},
    )
    assert set(absolute_dataset[0]) == {"target_beam"}

    with pytest.raises(TargetShotSupervisionGuardError, match="target_unlabeled.*beam_residual"):
        assert_target_supervision_allowed(
            {
                "subset": "target_unlabeled",
                "target_label_fraction": 0.05,
                "split_artifact_path": "split.json",
            },
            "beam_residual",
            scope="training",
        )
    assert_target_supervision_allowed({"subset": "target_labeled"}, "beam_residual", scope="training")
    assert_target_supervision_allowed({"subset": "target_test"}, "target_beam", scope="evaluation")


def test_target_shot_runtime_metadata_summary():
    artifact = build_target_shot_split(_target_shot_rows(), _default_cfg(), dataset_type="mmw")
    metadata = target_shot_runtime_metadata(artifact, artifact_path="split.json")
    assert metadata["source_domains"] == ["src:sunny"]
    assert metadata["target_domains"] == ["target:rain"]
    assert metadata["target_label_fraction"] == pytest.approx(0.05)
    assert metadata["target_labeled_count"] == 5
    assert metadata["strict_eligibility"]["eligible"] is True


class _AbsoluteProvider:
    target_schema = "absolute"

    def load(self, row):
        return {"target_beam": int(row.target_ref["beam_abs"])}

    def metadata(self):
        return {"target_schema": "absolute"}


def _default_cfg() -> TargetShotSplitConfig:
    return TargetShotSplitConfig(
        domain_type="scenario_weather",
        source_domains=("src:sunny",),
        target_domains=("target:rain",),
        target_label_fraction=0.05,
        target_label_selection="random",
        seed=123,
    )


def _target_shot_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(50):
        rows.append(
            {
                "sample_id": f"src-{idx}",
                "split": "train",
                "dataset_type": "mmw",
                "scenario": "src",
                "weather": "sunny",
                "beam_label": idx % 8,
                "beam_geo": (idx + 1) % 8,
                "beam_residual": 1,
                "geo_sector": idx % 4,
            }
        )
    for idx in range(100):
        rows.append(
            {
                "sample_id": f"target-adapt-{idx}",
                "split": "train",
                "dataset_type": "mmw",
                "scenario": "target",
                "weather": "rain",
                "beam_label": idx % 8,
                "beam_geo": (idx + 2) % 8,
                "beam_residual": 2,
                "geo_sector": idx % 4,
            }
        )
    for idx in range(20):
        rows.append(
            {
                "sample_id": f"target-test-{idx}",
                "split": "test",
                "dataset_type": "mmw",
                "scenario": "target",
                "weather": "rain",
                "beam_label": (idx + 3) % 8,
                "beam_geo": (idx + 1) % 8,
                "beam_residual": -1,
                "geo_sector": idx % 4,
            }
        )
    return rows
