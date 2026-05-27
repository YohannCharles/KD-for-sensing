from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.cli.hist_beam_loso import build_loso_run_plan, main as hist_beam_loso_main, run_hist_beam_loso  # noqa: E402
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.layouts import deepsense6g_scene_layout  # noqa: E402
from kd_sensing.data.loso import (  # noqa: E402
    default_loso_folds,
    resolve_loso_fold,
    sample_few_shot_records,
    split_target_records,
)
from kd_sensing.engine.hist_beam_adaptation import (  # noqa: E402
    apply_hist_beam_adaptation_strategy,
    trainable_parameter_summary,
)
from kd_sensing.engine.hist_beam_labels import hist_beam_labels  # noqa: E402
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, prototype_consistency_loss  # noqa: E402
from kd_sensing.engine.hist_beam_prototypes import (  # noqa: E402
    load_source_prototypes,
    prototype_coverage_from_counts,
    validate_prototype_artifact,
)
from kd_sensing.engine.hist_beam_loso_execution import (  # noqa: E402
    SOURCE_ONLY_VARIANTS,
    _few_shot_adaptation_loaders,
    run_loso_execute_preflight,
)
from kd_sensing.engine.loso_data import build_target_adapt_test_datasets  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import (  # noqa: E402
    beam_power_metrics,
    calculate_hist_beam_metrics,
    summarize_loso_runs,
    write_hist_beam_predictions,
)
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402
from kd_sensing.registries import MODELS, import_default_components  # noqa: E402


def test_deepsense_scenario_33_34_config_and_layout_support():
    scene33 = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=scenario33"])
    scene34 = load_config(ROOT / "configs/mmwave/teacher_no_kd.yaml", ["data.dataset.scene=34"])

    assert scene33["data"]["dataset"]["scene_id"] == 33
    assert scene33["data"]["dataset"]["scene_slug"] == "scene33"
    assert scene33["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario33"
    assert scene34["data"]["dataset"]["scene_id"] == 34
    assert scene34["data"]["dataset"]["scene_slug"] == "scene34"
    assert scene34["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario34"
    assert deepsense6g_scene_layout(34).legacy_root == "dataset/scenario34"


@pytest.mark.parametrize(("removed_type", "scene"), [("scenario33", 33), ("scenario34", 34)])
def test_deepsense_scenario_33_34_removed_dataset_types_are_rejected(removed_type: str, scene: int):
    with pytest.raises(ValueError, match=f"deepsense6g.*scene: {scene}"):
        load_config(
            ROOT / "configs/mmwave/teacher_no_kd.yaml",
            [f"data.dataset.type={removed_type}", "data.dataset.scene=null"],
        )


def test_loso_folds_and_explicit_overlap_rejection():
    folds = default_loso_folds()

    assert [fold.target_scene for fold in folds] == [34, 33, 32, 31]
    assert folds[0].source_scenes == (31, 32, 33)
    assert resolve_loso_fold(target_scene=34).metadata()["fold_id"] == "target_scene34"

    with pytest.raises(ValueError, match="source/target scene must not overlap"):
        resolve_loso_fold(target_scene=34, source_scenes=[31, 34])


def test_target_split_prefers_complete_seq_index_and_is_reproducible():
    records = [
        {"sample_id": f"s{i}", "seq_index": i // 2, "beam": i % 8}
        for i in range(10)
    ]

    split_a = split_target_records(records, adapt_fraction=0.4, seed=7)
    split_b = split_target_records(records, adapt_fraction=0.4, seed=7)

    assert split_a.adapt_indices == split_b.adapt_indices
    assert set(split_a.adapt_indices).isdisjoint(split_a.test_indices)
    assert not set(split_a.metadata["adapt_seq_index"]) & set(split_a.metadata["test_seq_index"])
    assert split_a.metadata["target_adapt_count"] + split_a.metadata["target_test_count"] == len(records)


def test_target_split_uses_post_portion_dataset_indices(tmp_path: Path):
    root = tmp_path / "scenario34"
    _write_many_scene_fixture(root, row_count=100)
    cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")
    cfg["loso"]["scene_data_roots"] = {"34": str(root)}
    cfg["data"]["dataset"]["test_csv_name"] = "test.csv"
    cfg["data"]["dataset"]["seq_len"] = 1
    cfg["data"]["dataset"]["num_pred"] = 1
    cfg["data"]["dataset"]["portion"] = 0.05
    cfg["data"]["dataset"]["use_gps"] = False
    cfg["model"]["modalities"] = ["image", "radar"]
    cfg["model"]["teacher"]["modalities"] = ["image", "radar"]
    cfg["model"]["student"]["modalities"] = ["image", "radar"]

    target_adapt, target_test, split = build_target_adapt_test_datasets(
        cfg,
        target_scene=34,
        source_scenes=[31, 32, 33],
        adapt_fraction=0.4,
        split_seed=7,
    )

    local_indices = set(target_adapt.indices) | set(target_test.indices)
    csv_indices = set(target_adapt.csv_indices) | set(target_test.csv_indices)
    assert len(target_adapt) > 0
    assert len(target_test) > 0
    assert local_indices == set(range(5))
    assert csv_indices == {0, 25, 50, 74, 99}
    assert split.metadata["target_dataset_count"] == 5
    assert split.metadata["target_selection"]["selected_rows"] == 5


def test_few_shot_sampler_handles_zero_budget_stratification_and_degrade():
    records = [{"sample_id": f"s{i}", "beam": i} for i in range(16)]

    zero = sample_few_shot_records(records, budget=0, seed=1, group_size=4)
    few = sample_few_shot_records(records, budget=5, seed=1, group_size=4)
    degrade = sample_few_shot_records(records[:3], budget=5, seed=1, group_size=4)

    assert zero.labeled_indices == ()
    assert zero.manifest["unlabeled_count"] == 16
    assert few.manifest["actual_labeled_count"] == 5
    assert len({item["coarse_group"] for item in few.manifest["labeled_samples"]}) >= 4
    assert degrade.labeled_indices == (0, 1, 2)
    assert degrade.manifest["degrade_reason"] == "requested_budget_exceeds_available_target_adapt"


def test_few_shot_sampler_resolves_explicit_and_power_path_labels(tmp_path: Path):
    root = tmp_path / "scenario34"
    root.mkdir()
    _write_power_vector(root / "future_a.txt", label=9)
    _write_power_vector(root / "future_b.txt", label=24)
    records = [
        {"sample_id": "explicit", "future_beam1": "/unused.txt", "future_beam_label1": "17"},
        {"sample_id": "path_a", "future_beam1": "/future_a.txt"},
        {"sample_id": "path_b", "future_beam1": "/future_b.txt"},
    ]

    sampled = sample_few_shot_records(
        records,
        budget=5,
        seed=0,
        group_size=8,
        label_key="future_beam1",
        data_root=root,
    )

    by_id = {item["sample_id"]: item for item in sampled.manifest["labeled_samples"]}
    assert sampled.manifest["degrade_reason"] == "requested_budget_exceeds_available_target_adapt"
    assert by_id["explicit"]["beam"] == 17
    assert by_id["explicit"]["label_source"] == "future_beam_label1"
    assert by_id["path_a"]["beam"] == 9
    assert by_id["path_a"]["label_source"] == "future_beam1:power_argmax"
    assert by_id["path_b"]["coarse_group"] == 3


def test_loso_few_shot_loader_handles_power_path_labels(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    target_adapt, _, _ = build_target_adapt_test_datasets(
        cfg,
        target_scene=34,
        source_scenes=[31, 32, 33],
        split_seed=0,
    )

    labeled_loader, unlabeled_loader, manifest = _few_shot_adaptation_loaders(
        target_adapt,
        cfg,
        {"budget": 10, "seed": 0},
        loader_kwargs={"batch_size": 1, "shuffle": False, "num_workers": 0},
    )

    assert labeled_loader is not None
    assert unlabeled_loader is None
    assert manifest["actual_labeled_count"] == len(target_adapt)
    assert manifest["labeled_samples"][0]["beam"] == 3
    assert manifest["labeled_samples"][0]["label_source"] == "future_beam1:power_argmax"


def test_hist_beam_registry_forward_shapes_loss_and_adapter_equivalence():
    import_default_components()
    model = MODELS.build(
        {
            "type": "hist_beam_fusion",
            "modalities": ["image", "radar", "gps"],
            "feature_size": 8,
            "d_model": 16,
            "num_classes": 16,
            "num_pred": 2,
            "group_size": 4,
            "variant": "v4_adapter",
            "num_heads": 4,
            "num_layers": 1,
            "image_encoder": {"type": "legacy_cnn"},
        }
    )
    assert isinstance(model, HistBeamFusionNet)
    model.eval()

    with torch.no_grad():
        output = model(
            image_batch=torch.randn(2, 3, 3, 224, 224),
            radar_batch=torch.randn(2, 3, 2, 128, 64),
            gps_batch=torch.randn(2, 3, 3),
        )

    assert output["logits"].shape == (2, 2, 16)
    assert output["coarse_logits"].shape == (2, 2, 4)
    assert output["fine_logits"].shape == (2, 2, 4, 4)
    assert output["shared_representation"].shape == (2, 2, 16)
    assert torch.allclose(output["private_representation"], output["adapter_representation"])

    labels = torch.tensor([[0, 7], [12, 15]])
    loss = compute_hist_beam_loss(output, labels, cfg={"hist_beam": {"group_size": 4}})
    assert loss.total.isfinite()
    assert loss.diagnostics["hist/loss_coarse"] > 0.0


def test_hist_beam_label_helper_rejects_invalid_group_size():
    labels = torch.tensor([[0, 7], [8, 15]])

    coarse, fine = hist_beam_labels(labels, num_classes=16, group_size=4)

    assert coarse.tolist() == [[0, 1], [2, 3]]
    assert fine.tolist() == [[0, 3], [0, 3]]
    with pytest.raises(ValueError, match="divisible"):
        hist_beam_labels(labels, num_classes=10, group_size=4)


def test_adaptation_freezing_and_full_finetune_parameter_ratios():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v4_adapter",
        num_heads=4,
        num_layers=1,
    )

    adapter = apply_hist_beam_adaptation_strategy(model, "v4_adapter")
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    full = apply_hist_beam_adaptation_strategy(model, "v6_full_finetune")

    assert adapter["trainable_ratio"] < 1.0
    assert any(name.startswith("private_adapter") for name in trainable_names)
    assert any(name.startswith("fine_head") for name in trainable_names)
    assert full["trainable_ratio"] == pytest.approx(1.0)
    assert trainable_parameter_summary(model).ratio == pytest.approx(1.0)


def test_prototype_artifact_loading_coverage_and_confidence_filter(tmp_path: Path):
    artifact = {
        "version": "hist_beam_prototypes_v1",
        "shared_prototypes": torch.eye(3, 4),
        "private_prototypes": torch.eye(3, 4),
        "counts": torch.tensor([2, 0, 5]),
        "metadata": {"group_size": 4, "num_groups": 3},
    }
    path = tmp_path / "proto.pt"
    torch.save(artifact, path)

    loaded = load_source_prototypes(path)
    validate_prototype_artifact(loaded)
    coverage = prototype_coverage_from_counts(loaded["counts"])
    loss, metrics = prototype_consistency_loss(torch.randn(4, 4), loaded["shared_prototypes"], confidence_threshold=0.99)

    assert coverage["empty_groups"] == [1]
    assert coverage["prototype_coverage"] == pytest.approx(2 / 3)
    assert loss.isfinite()
    assert 0.0 <= metrics["prototype_coverage"] <= 1.0


def test_hist_beam_metrics_prediction_writer_power_and_summary(tmp_path: Path):
    outputs = torch.tensor([[[3.0, 1.0, 0.0, -1.0]], [[0.0, 1.0, 4.0, -1.0]]])
    labels = torch.tensor([[0], [2]])
    metrics = calculate_hist_beam_metrics(outputs, labels, group_size=2, num_classes=4)
    pred_path = write_hist_beam_predictions(
        tmp_path / "predictions.csv",
        outputs,
        labels,
        metadata=[{"sample_id": "a", "scene_slug": "scene31"}, {"sample_id": "b", "scene_slug": "scene32"}],
        group_size=2,
        top_k=3,
        variant_metadata={"variant": "v3_decoupled"},
    )
    power_missing = beam_power_metrics(torch.tensor([0, 2]), torch.tensor([0, 2]), None)
    power = beam_power_metrics(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([[1.0, 0.5, 0.1], [0.2, 0.5, 1.0]]),
    )
    summary = summarize_loso_runs(
        [
            {"summary_type": "source_only", "fold": "f1", "target_scene": 31, "variant": "v3", "budget": 0, "seed": 0, "metrics": {"top1": 0.5}, "run_path": "a"},
            {"summary_type": "source_only", "fold": "f1", "target_scene": 31, "variant": "v3", "budget": 0, "seed": 1, "metrics": {"top1": 1.0}, "run_path": "b"},
        ]
    )

    assert metrics["coarse_accuracy"] == pytest.approx(1.0)
    assert pred_path.exists()
    with pred_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["sample_id"] == "a"
    assert json.loads(rows[0]["topk_predictions"]) == [0, 1, 2]
    assert rows[0]["predicted_beam"] == rows[0]["pred_beam"]
    assert rows[0]["split"] == "test"
    assert power_missing["power_metrics_available"] is False
    assert power["power_metrics_available"] is True
    assert summary["rows"][0]["top1_mean"] == pytest.approx(0.75)


def test_hist_beam_quick_smoke_is_resource_probe_and_quick_validation_can_expand():
    smoke_cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")
    validation_cfg = load_config(ROOT / "configs/hist_beam/quick_validation.yaml")

    plan = build_loso_run_plan(
        smoke_cfg,
        target_scenes=smoke_cfg["loso"]["target_scenes"],
        variants=smoke_cfg["loso"]["variants"],
        budgets=smoke_cfg["loso"]["budgets"],
        seeds=smoke_cfg["loso"]["seeds"],
        max_runs=smoke_cfg["loso"]["max_runs"],
    )
    validation_plan = build_loso_run_plan(
        validation_cfg,
        target_scenes=validation_cfg["loso"]["target_scenes"],
        variants=validation_cfg["loso"]["variants"],
        budgets=validation_cfg["loso"]["budgets"],
        seeds=validation_cfg["loso"]["seeds"],
    )
    expanded = build_loso_run_plan(
        validation_cfg,
        target_scenes=[34, 33, 32, 31],
        variants=["v3_decoupled"],
        budgets=[0],
        seeds=[0],
    )
    limited = build_loso_run_plan(
        validation_cfg,
        target_scenes=[34],
        variants=["v0_flat", "v3_decoupled", "v5_adapter_proto"],
        budgets=[0, 10],
        seeds=[0],
        max_runs=2,
    )
    skipped = build_loso_run_plan(validation_cfg, target_scenes=[34], skip_scenes=[34], variants=["v3_decoupled"], budgets=[0], seeds=[0])
    single = build_loso_run_plan(
        validation_cfg,
        target_scene=33,
        source_scenes=None,
        variants=["v3_decoupled"],
        budgets=[0],
        seeds=[0],
    )

    assert {run["target_scene"] for run in plan["runs"]} == {34}
    assert {run["variant"] for run in plan["runs"]} == {"v0_flat"}
    assert {run["budget"] for run in plan["runs"]} == {0}
    assert {run["seed"] for run in plan["runs"]} == {0}
    assert smoke_cfg["data"]["dataset"]["portion"] < 1.0
    assert smoke_cfg["training"]["epochs"] <= 2
    assert plan["planned_run_count"] == 1
    assert plan["max_runs"] == 1
    assert {run["variant"] for run in validation_plan["runs"]} == {
        "v0_flat",
        "v3_decoupled",
        "v4_adapter",
        "v5_adapter_proto",
        "v6_full_finetune",
    }
    assert {run["budget"] for run in validation_plan["runs"]} == {0, 10}
    assert {run["seed"] for run in validation_plan["runs"]} == {0}
    assert validation_cfg["data"]["dataset"]["portion"] == pytest.approx(1.0)
    assert validation_cfg["training"]["epochs"] == 40
    assert limited["planned_run_count"] == 6
    assert len(limited["runs"]) == 2
    assert limited["max_runs"] == 2
    assert {run["target_scene"] for run in expanded["runs"]} == {31, 32, 33, 34}
    assert skipped["runs"] == []
    assert single["runs"][0]["source_scenes"] == [31, 32, 34]
    assert all(run["target_test_for_training"] is False for run in validation_plan["runs"])


def test_hist_beam_loso_execute_uses_runner_and_records_stage_metadata(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    executor = MetadataAssertingStageExecutor()
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v3_decoupled", budgets="0"),
        stage_executor=executor,
    )

    execution = result["execution"]
    assert execution["status"] == "completed"
    assert execution["status"] != "planned"
    assert result["run_count"] == 1
    run = execution["runs"][0]
    assert [stage["name"] for stage in run["stages"]] == [
        "source_train",
        "source_only_target_test_eval",
        "target_adaptation",
        "adapted_target_test_eval",
        "summary",
    ]
    assert run["fold"] == "target_scene34"
    assert run["target_scene"] == 34
    assert run["source_scenes"] == [31, 32, 33]
    assert run["stages"][0]["artifacts"]["source_checkpoint_path"]
    assert (Path(run["artifacts"]["run_metadata_path"])).exists()
    progress_events = (tmp_path / "out" / "execution_progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["event"] == "stage_started" for line in progress_events)


def test_hist_beam_loso_execute_preflight_fails_before_stages_for_missing_data(tmp_path: Path):
    cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")
    cfg["loso"]["scene_data_roots"] = {str(scene): str(tmp_path / f"missing{scene}") for scene in (31, 32, 33, 34)}
    plan = build_loso_run_plan(cfg, target_scenes=[34], variants=["v3_decoupled"], budgets=[0], seeds=[0])

    preflight = run_loso_execute_preflight(plan, cfg, tmp_path / "out")

    assert preflight["status"] == "failed"
    assert any(error["resource_type"] == "data_root" and error["scene"] == 31 for error in preflight["errors"])
    assert any("missing" in str(error["path"]) for error in preflight["errors"])


def test_hist_beam_loso_execute_smoke_writes_summary_and_conclusion(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v3_decoupled,v5_adapter_proto,v6_full_finetune", budgets="0"),
        stage_executor=FakeStageExecutor(),
    )

    execution = result["execution"]
    summary_json = Path(execution["summary_paths"]["json"])
    summary_csv = Path(execution["summary_paths"]["csv"])
    conclusion = Path(execution["summary_paths"]["quick_validation_conclusion"])

    assert execution["status"] == "completed"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert conclusion.exists()
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with conclusion.open("r", encoding="utf-8") as f:
        conclusion_payload = json.load(f)
    assert summary["run_count"] == 3
    assert any(row["variant"] == "v5_adapter_proto" for row in summary["runs"])
    assert conclusion_payload["summary_path"] == str(summary_json)


def test_hist_beam_loso_execute_keyboard_interrupt_writes_partial_summary(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v3_decoupled,v5_adapter_proto", budgets="0"),
        stage_executor=InterruptingStageExecutor(),
    )

    execution = result["execution"]
    summary_json = Path(execution["summary_paths"]["json"])
    conclusion = Path(execution["summary_paths"]["quick_validation_conclusion"])

    assert execution["status"] == "partial_failed"
    assert execution["interrupted"] is True
    assert summary_json.exists()
    assert conclusion.exists()
    assert [run["status"] for run in execution["runs"]] == ["failed", "missing"]
    assert "KeyboardInterrupt" in execution["runs"][0]["failure_reason"]
    with summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["runs"][1]["run_status"] == "missing"
    progress_events = (tmp_path / "out" / "execution_progress.jsonl").read_text(encoding="utf-8")
    assert "stage_interrupted" in progress_events


def test_hist_beam_loso_console_main_returns_zero_for_successful_plan(tmp_path: Path):
    exit_code = hist_beam_loso_main(
        [
            "--config",
            str(ROOT / "configs/hist_beam/quick_smoke.yaml"),
            "--target-scene",
            "34",
            "--variants",
            "v3_decoupled",
            "--budgets",
            "0",
            "--seeds",
            "0",
            "--output-dir",
            str(tmp_path / "plan"),
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "plan" / "loso_plan.json").exists()


class FakeStageExecutor:
    def execute(self, stage: str, run, context):
        context.stage_dir.mkdir(parents=True, exist_ok=True)
        variant = str(run["variant"])
        if stage == "source_train":
            checkpoint = context.stage_dir / "source_checkpoint.pth"
            prototype = context.stage_dir / "source_prototypes.pt"
            metrics = context.stage_dir / "metrics.json"
            checkpoint.write_text("fake checkpoint", encoding="utf-8")
            prototype.write_text("fake prototypes", encoding="utf-8")
            _write_json(metrics, {"train_loss_last": 0.1})
            return {
                "status": "completed",
                "artifacts": {
                    "source_checkpoint_path": str(checkpoint),
                    "source_prototype_path": str(prototype),
                    "metrics_path": str(metrics),
                },
                "metrics": {"train_loss_last": 0.1},
                "checkpoint_reuse": {"enabled": True, "reused": False},
            }
        if stage == "source_only_target_test_eval":
            return self._evaluation_result(context, run, summary_type="source_only", top1=0.4)
        if stage == "target_adaptation":
            if variant in SOURCE_ONLY_VARIANTS:
                return {"status": "skipped", "metrics": {"prototype_coverage_unavailable_reason": "source_only_variant"}}
            checkpoint = context.stage_dir / "adaptation_checkpoint.pth"
            metrics = context.stage_dir / "metrics.json"
            checkpoint.write_text("fake adaptation", encoding="utf-8")
            payload = {
                "trainable_params": 10 if variant != "v6_full_finetune" else 100,
                "total_params": 100,
                "trainable_ratio": 0.1 if variant != "v6_full_finetune" else 1.0,
                "adaptation_time_seconds": 1.0 if variant != "v6_full_finetune" else 3.0,
                "adaptation_time_per_epoch": 1.0 if variant != "v6_full_finetune" else 3.0,
                "prototype_coverage": 0.75 if variant == "v5_adapter_proto" else None,
                "prototype_coverage_unavailable_reason": None if variant == "v5_adapter_proto" else "variant_without_prototype_alignment",
            }
            _write_json(metrics, payload)
            return {
                "status": "completed",
                "artifacts": {"adaptation_checkpoint_path": str(checkpoint), "metrics_path": str(metrics)},
                "metrics": payload,
            }
        if stage == "adapted_target_test_eval":
            if variant in SOURCE_ONLY_VARIANTS:
                return {"status": "skipped", "metrics": {}}
            top1 = 0.62 if variant == "v5_adapter_proto" else 0.58
            if variant == "v6_full_finetune":
                top1 = 0.6
            return self._evaluation_result(context, run, summary_type="adapted", top1=top1)
        raise AssertionError(stage)

    @staticmethod
    def _evaluation_result(context, run, *, summary_type: str, top1: float):
        metrics = context.stage_dir / "metrics.json"
        predictions = context.stage_dir / "predictions.csv"
        payload = {
            "top1": top1,
            "top3": top1 + 0.1,
            "top5": top1 + 0.2,
            "coarse_accuracy": top1 + 0.05,
            "fine_offset_accuracy": top1 - 0.05,
            "summary_type": summary_type,
        }
        _write_json(metrics, payload)
        predictions.write_text(
            "sample_id,scene,true_beam,predicted_beam,topk_predictions,coarse_true,coarse_pred,fine_true,fine_pred,split,variant_metadata\n"
            f"s,{run['target_scene']},1,1,\"[1, 2, 3]\",0,0,1,1,target_test,\"{{}}\"\n",
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "artifacts": {"metrics_path": str(metrics), "predictions_path": str(predictions)},
            "metrics": payload,
        }


class MetadataAssertingStageExecutor(FakeStageExecutor):
    def execute(self, stage: str, run, context):
        with (context.run_dir / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["stages"][-1]["name"] == stage
        assert metadata["stages"][-1]["status"] == "running"
        return super().execute(stage, run, context)


class InterruptingStageExecutor:
    def execute(self, stage: str, run, context):
        raise KeyboardInterrupt()


def _loso_args(tmp_path: Path, *, execute: bool, variants: str = "v3_decoupled", budgets: str = "0"):
    return SimpleNamespace(
        target_scene=None,
        source_scenes=None,
        skip_scenes=None,
        variants=variants,
        budgets=budgets,
        seeds="0",
        output_dir=str(tmp_path / "out"),
        overwrite=True,
        resume=False,
        execute=execute,
    )


def _loso_fixture_config(tmp_path: Path) -> dict:
    cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")
    roots = {}
    for scene in (31, 32, 33, 34):
        root = tmp_path / f"scenario{scene}"
        _write_scene_fixture(root)
        roots[str(scene)] = str(root)
    cfg["loso"]["scene_data_roots"] = roots
    cfg["data"]["dataset"]["train_csv_name"] = "train.csv"
    cfg["data"]["dataset"]["test_csv_name"] = "test.csv"
    cfg["data"]["dataset"]["seq_len"] = 2
    cfg["data"]["dataset"]["num_pred"] = 1
    cfg["loso"]["max_runs"] = None
    return cfg


def _write_scene_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    headers = [
        "seq_index",
        "beam1",
        "beam2",
        "future_beam1",
        "camera1",
        "camera2",
        "radar1",
        "radar2",
        "gps1",
        "gps2",
        "bs_gps1",
        "bs_gps2",
    ]
    row = {
        "seq_index": "0",
        "beam1": "/beam1.txt",
        "beam2": "/beam2.txt",
        "future_beam1": "/future_beam1.txt",
        "camera1": "/camera1.jpg",
        "camera2": "/camera2.jpg",
        "radar1": "/radar1_RA.npy",
        "radar2": "/radar2_RA.npy",
        "gps1": "/gps1.txt",
        "gps2": "/gps2.txt",
        "bs_gps1": "/bs_gps1.txt",
        "bs_gps2": "/bs_gps2.txt",
    }
    for name in row.values():
        if name != "0":
            path = root / str(name).lstrip("/")
            if "beam" in path.name:
                _write_power_vector(path, label=3)
            else:
                path.write_text("0\n", encoding="utf-8")
    for radar_name in ("radar1_DA.npy", "radar2_DA.npy"):
        (root / radar_name).write_text("0\n", encoding="utf-8")
    for csv_name in ("train.csv", "test.csv"):
        with (root / csv_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)


def _write_many_scene_fixture(root: Path, *, row_count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    headers = ["seq_index", "beam1", "future_beam1", "camera1", "radar1"]
    with (root / "test.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "seq_index": str(index),
                    "beam1": f"/beam_{index}.txt",
                    "future_beam1": f"/future_{index}.txt",
                    "camera1": f"/camera_{index}.jpg",
                    "radar1": f"/radar_{index}_RA.npy",
                }
            )


def _write_power_vector(path: Path, *, label: int, num_classes: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = ["0"] * int(num_classes)
    values[int(label)] = "1"
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
