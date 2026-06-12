from __future__ import annotations

from pathlib import Path

from kd_sensing.utils.runtime_output_layout import (
    canonical_runtime_partitions,
    evaluation_output_base,
    output_layout_summary,
    runtime_output_scope_from_config,
    scenegroup_slug,
    scoped_output_base,
)


def test_scenegroup_slug_handles_contiguous_and_sparse_scenes():
    assert scenegroup_slug([32, 33, 34]) == "scenegroup_s32_s34"
    assert scenegroup_slug([31, 32, 33, 34]) == "scenegroup_s31_s34"
    assert scenegroup_slug([31, 33]) == "scenegroup_s31_s33"
    assert scenegroup_slug(["scene31", "scene33", "scene34"]) == "scenegroup_s31_s33_s34"


def test_runtime_output_scope_infers_single_and_multiscene_roles():
    single = runtime_output_scope_from_config({"data": {"dataset": {"type": "deepsense6g"}}})
    train_group = runtime_output_scope_from_config(
        {"data": {"dataset": {"type": "deepsense6g", "train_scenes": [32, 33, 34]}}}
    )
    eval_group = runtime_output_scope_from_config(
        {"data": {"dataset": {"type": "deepsense6g", "eval_scenes": [31, 32, 33, 34]}}},
        purpose="evaluation",
    )

    assert single is not None
    assert single.kind == "scene"
    assert single.slug == "scene31"
    assert train_group is not None
    assert train_group.kind == "scenegroup"
    assert train_group.slug == "scenegroup_s32_s34"
    assert train_group.to_metadata()["train_scenes"] == [32, 33, 34]
    assert eval_group is not None
    assert eval_group.slug == "scenegroup_s31_s34"
    assert eval_group.to_metadata()["eval_scenes"] == [31, 32, 33, 34]


def test_scoped_output_base_and_evaluation_base_respect_explicit_overrides(tmp_path: Path):
    cfg = {
        "experiment": {"name": "study"},
        "data": {"dataset": {"type": "deepsense6g", "train_scenes": [32, 34]}},
        "output": {"group_by_scene": True},
    }
    explicit_cfg = {**cfg, "output": {"group_by_scene": False}}

    assert scoped_output_base(tmp_path / "outputs", cfg) == tmp_path / "outputs" / "scenegroup_s32_s34"
    assert scoped_output_base(tmp_path / "manual_run", explicit_cfg) == tmp_path / "manual_run"
    assert evaluation_output_base(tmp_path / "outputs", cfg) == tmp_path / "outputs" / "evaluations" / "study"


def test_canonical_partition_summary_marks_scope_and_legacy_paths(tmp_path: Path):
    partitions = canonical_runtime_partitions(tmp_path / "outputs")
    assert partitions["cache"].endswith("outputs/cache")
    assert partitions["scenegroup"].endswith("outputs/scenegroup_<range-or-list>")

    scene_run = tmp_path / "outputs" / "scene31" / "run"
    group_run = tmp_path / "outputs" / "scenegroup_s32_s34" / "run"
    numeric = tmp_path / "outputs" / "31"
    registry = tmp_path / "outputs" / "best_checkpoints"

    assert output_layout_summary(scene_run)["canonical_partition"] == "scene"
    assert output_layout_summary(scene_run)["scope_slug"] == "scene31"
    assert output_layout_summary(group_run)["canonical_partition"] == "scenegroup"
    assert output_layout_summary(numeric)["canonical_partition"] == "legacy_numeric_scene"
    assert output_layout_summary(numeric)["legacy"] is True
    assert output_layout_summary(registry)["canonical_partition"] == "legacy_registry"
