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
    TargetPrivatePrototypeBank,
    adapt_hist_beam_target,
    apply_hist_beam_adaptation_strategy,
    radio_prototype_assignment,
    trainable_parameter_summary,
)
from kd_sensing.engine.hist_beam_labels import hist_beam_labels  # noqa: E402
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss, prototype_consistency_loss  # noqa: E402
from kd_sensing.engine.hist_beam_prototypes import (  # noqa: E402
    generate_source_prototypes,
    load_source_prototypes,
    prototype_coverage_from_counts,
    validate_prototype_artifact,
)
from kd_sensing.engine.hist_beam_loso_execution import (  # noqa: E402
    SOURCE_ONLY_VARIANTS,
    _few_shot_adaptation_loaders,
    _prototype_decision,
    _stage_cfg,
    run_loso_execute_preflight,
    write_loso_execute_summary,
    write_quick_validation_conclusion,
)
from kd_sensing.engine.hist_beam_loso_config import _source_variant_for  # noqa: E402
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities  # noqa: E402
from kd_sensing.engine.runtime import prepare_task_inputs  # noqa: E402
import kd_sensing.engine.loso_data as loso_data_module  # noqa: E402
from kd_sensing.engine.loso_data import (  # noqa: E402
    build_loso_source_train_loader,
    build_loso_target_stage_loader,
    build_target_adapt_test_datasets,
)
from kd_sensing.evaluation.hist_beam_outputs import (  # noqa: E402
    beam_power_metrics,
    calculate_hist_beam_metrics,
    radio_semantic_metrics,
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


def test_few_shot_sampler_prefers_radio_semantic_stratification():
    records = [
        {"sample_id": f"s{i}", "beam": i, "radio_semantic_label": i % 3, "relative_azimuth_bin": i % 2}
        for i in range(9)
    ]

    sampled = sample_few_shot_records(records, budget=5, seed=2, group_size=4)

    assert sampled.manifest["stratification"] == "radio_semantic"
    assert sampled.manifest["protocol"] == "radio_semantic_stratified_few_shot"
    assert sampled.manifest["radio_stratification_unavailable_reason"] is None
    assert len({item["radio_semantic_label"] for item in sampled.manifest["labeled_samples"]}) == 3


def test_few_shot_sampler_can_use_beam_frequency_stratification():
    beams = [48] * 6 + [50] * 5 + [47] * 4 + [49] * 3 + [34] * 2 + [1]
    records = [
        {"sample_id": f"s{i}", "beam": beam, "radio_semantic_label": i % 2, "relative_azimuth_bin": i % 3}
        for i, beam in enumerate(beams)
    ]

    sampled = sample_few_shot_records(
        records,
        budget=5,
        seed=2,
        group_size=8,
        stratification="beam_frequency",
    )

    sampled_beams = [item["beam"] for item in sampled.manifest["labeled_samples"]]
    assert sampled.manifest["stratification"] == "beam_frequency"
    assert sampled.manifest["protocol"] == "beam_frequency_stratified_few_shot"
    assert set(sampled_beams[:4]) == {47, 48, 49, 50}


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


def test_hist_beam_radio_forward_condition_and_loss():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=2,
        group_size=4,
        variant="v6_radio_proto",
        radio_semantic={"enabled": True, "num_spread_bins": 3, "use_radio_head": True, "use_radio_condition_in_beam_head": True},
        num_radio_classes=12,
        use_radio_head=True,
        use_radio_condition_in_beam_head=True,
        radio_embed_dim=6,
        num_heads=4,
        num_layers=1,
    )
    output = model(gps_batch=torch.randn(3, 2, 3))
    labels = torch.tensor([[0, 7], [12, 15], [4, 8]])
    radio = torch.tensor([[0, 5], [11, -100], [2, 7]])
    loss = compute_hist_beam_loss(
        output,
        labels,
        cfg={"hist_beam": {"group_size": 4, "loss_weights": {"radio_semantic": 0.5}}},
        radio_semantic_labels=radio,
    )

    assert output["logits"].shape == (3, 2, 16)
    assert output["radio_logits"].shape == (3, 2, 12)
    assert output["radio_assignment"].shape == (3, 12)
    assert output["hist_beam"]["proto_type"] == "radio_semantic"
    assert loss.radio_semantic.isfinite()
    assert loss.diagnostics["hist/radio_coverage"] > 0.0


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


def test_radio_prototype_assignment_bank_and_artifact_generation(tmp_path: Path):
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v6_radio_proto",
        radio_semantic={"enabled": True, "use_radio_head": True},
        num_radio_classes=12,
        use_radio_head=True,
        num_heads=4,
        num_layers=1,
    )
    batch = {
        "gps": torch.randn(4, 1, 3),
        "input_beam": torch.tensor([[0], [1], [2], [3]]),
        "target_beam": torch.tensor([[0], [1], [2], [3]]),
        "radio_semantic_label": torch.tensor([[0], [1], [1], [5]]),
        "radio_semantic_available": torch.ones(4, 1, dtype=torch.bool),
    }
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {
            "seq_length_student": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {
                "type": "hist_beam_fusion",
                "modalities": ["gps"],
                "num_classes": 16,
                "group_size": 4,
                "num_radio_classes": 12,
                "d_model": 16,
                "variant": "v6_radio_proto",
            },
        },
        "hist_beam": {
            "group_size": 4,
            "proto_type": "radio_semantic",
            "radio_semantic": {"enabled": True, "num_radio_classes": 12},
        },
    }
    artifact = generate_source_prototypes(
        model,
        torch.utils.data.DataLoader([batch], batch_size=None),
        cfg,
        torch.device("cpu"),
        output_path=tmp_path / "radio_proto.pt",
    )
    alpha, metrics = radio_prototype_assignment(
        torch.randn(4, 16),
        artifact["mu_radio_c"],
        counts=artifact["count_radio"],
        tau=1.0,
    )
    bank = TargetPrivatePrototypeBank(num_classes=12, dim=16, device=torch.device("cpu"), dtype=torch.float32)
    update = bank.update(torch.randn(4, 16), alpha.argmax(dim=-1), alpha.max(dim=-1).values, threshold=0.0)
    bank_loss, bank_metrics = bank.loss(torch.randn(4, 16), alpha.argmax(dim=-1))

    validate_prototype_artifact(artifact)
    assert artifact["metadata"]["prototype_space"] == "shared_radio_semantic"
    assert artifact["count_radio"][1].item() == 2
    assert metrics["radio_prototype_available_classes"] >= 3
    assert update["target_private_update_used"] == 4
    assert bank_loss.isfinite()
    assert bank_metrics["target_private_prototype_used"] > 0


def test_zero_label_radio_adaptation_records_no_target_leakage():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v6_radio_proto",
        radio_semantic={"enabled": True, "use_radio_head": True},
        num_radio_classes=12,
        use_radio_head=True,
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v6_radio_proto")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    batch = {
        "gps": torch.randn(2, 1, 3),
        "input_beam": torch.tensor([[0], [1]]),
        "target_beam": torch.tensor([[0], [1]]),
        "radio_semantic_label": torch.tensor([[0], [1]]),
        "radio_semantic_available": torch.ones(2, 1, dtype=torch.bool),
    }
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {
            "seq_length_student": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {
                "type": "hist_beam_fusion",
                "modalities": ["gps"],
                "num_classes": 16,
                "group_size": 4,
            },
        },
        "hist_beam": {"group_size": 4, "adaptation": {"entropy_weight": 0.01, "prototype_weight": 0.0}},
    }
    result = adapt_hist_beam_target(
        model,
        None,
        torch.utils.data.DataLoader([batch], batch_size=None),
        cfg,
        torch.device("cpu"),
        optimizer,
        epochs=1,
        label_budget=0,
    )

    assert result["used_target_labels"] is False
    assert result["used_target_beam_power_for_training"] is False
    assert result["used_target_radio_label_for_training"] is False
    assert result["sensitive_field_policy"]["target_unlabeled"]["radio"] == "blocked"
    assert result["main_conclusion_eligible"] is True


def test_labeled_target_radio_supervision_requires_opt_in_and_marks_ineligible():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v6_radio_proto",
        radio_semantic={"enabled": True, "use_radio_head": True},
        num_radio_classes=12,
        use_radio_head=True,
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v6_radio_proto")
    batch = {
        "gps": torch.randn(2, 1, 3),
        "input_beam": torch.tensor([[0], [1]]),
        "target_beam": torch.tensor([[0], [1]]),
        "radio_semantic_label": torch.tensor([[0], [1]]),
        "radio_semantic_available": torch.ones(2, 1, dtype=torch.bool),
    }
    cfg = _radio_adaptation_cfg(allow_radio=False)
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)

    with pytest.raises(RuntimeError, match="radio_semantic_label.*label_budget=2.*labeled_subset=True"):
        adapt_hist_beam_target(
            model,
            torch.utils.data.DataLoader([batch], batch_size=None),
            None,
            cfg,
            torch.device("cpu"),
            optimizer,
            epochs=1,
            label_budget=2,
        )

    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v6_radio_proto",
        radio_semantic={"enabled": True, "use_radio_head": True},
        num_radio_classes=12,
        use_radio_head=True,
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v6_radio_proto")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    result = adapt_hist_beam_target(
        model,
        torch.utils.data.DataLoader([batch], batch_size=None),
        None,
        _radio_adaptation_cfg(allow_radio=True),
        torch.device("cpu"),
        optimizer,
        epochs=1,
        label_budget=2,
    )

    assert result["used_target_beam_for_training"] is True
    assert result["used_target_radio_label_for_training"] is True
    assert result["main_conclusion_eligible"] is False
    assert "target_radio_label_supervision" in result["eligibility_reasons"]
    assert result["sensitive_field_policy"]["allow_labeled_target_radio_supervision"] is True


def test_labeled_target_path_supervision_requires_opt_in():
    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v8_path_proto",
        path_semantic={"enabled": True, "use_path_head": True, "num_path_classes": 4, "use_path_regression": True},
        num_path_classes=4,
        use_path_head=True,
        path_embed_dim=8,
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v8_path_proto")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    batch = {
        "gps": torch.randn(2, 1, 3),
        "input_beam": torch.tensor([[0], [1]]),
        "target_beam": torch.tensor([[0], [1]]),
        "path_semantic_label": torch.tensor([[0], [1]]),
        "path_descriptor": torch.randn(2, 1, 8),
        "path_valid": torch.ones(2, 1, dtype=torch.bool),
    }
    cfg = _path_adaptation_cfg(allow_path=False)

    with pytest.raises(RuntimeError, match="path_semantic_label.*label_budget=2.*labeled_subset=True"):
        adapt_hist_beam_target(
            model,
            torch.utils.data.DataLoader([batch], batch_size=None),
            None,
            cfg,
            torch.device("cpu"),
            optimizer,
            epochs=1,
            label_budget=2,
        )

    model = HistBeamFusionNet(
        modalities=["gps"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=1,
        group_size=4,
        variant="v8_path_proto",
        path_semantic={"enabled": True, "use_path_head": True, "num_path_classes": 4, "use_path_regression": True},
        num_path_classes=4,
        use_path_head=True,
        path_embed_dim=8,
        num_heads=4,
        num_layers=1,
    )
    apply_hist_beam_adaptation_strategy(model, "v8_path_proto")
    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.01)
    result = adapt_hist_beam_target(
        model,
        torch.utils.data.DataLoader([batch], batch_size=None),
        None,
        _path_adaptation_cfg(allow_path=True),
        torch.device("cpu"),
        optimizer,
        epochs=1,
        label_budget=2,
    )

    assert result["used_target_path_label_for_training"] is True
    assert result["used_target_path_descriptor_for_training"] is True
    assert result["main_conclusion_eligible"] is False
    assert "target_path_label_supervision" in result["eligibility_reasons"]
    assert "target_path_descriptor_supervision" in result["eligibility_reasons"]


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
        variant_metadata={"variant": "v1_hierarchical"},
        radio_logits=torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]]),
        radio_labels=torch.tensor([[0], [1]]),
    )
    power_missing = beam_power_metrics(torch.tensor([0, 2]), torch.tensor([0, 2]), None)
    power = beam_power_metrics(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([[1.0, 0.5, 0.1], [0.2, 0.5, 1.0]]),
    )
    radio_metrics = radio_semantic_metrics(
        torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]]),
        torch.tensor([[0], [1]]),
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
    assert rows[0]["radio_true"] == "0"
    assert rows[0]["radio_pred"] == "0"
    assert rows[0]["split"] == "test"
    assert power_missing["power_metrics_available"] is False
    assert power["power_metrics_available"] is True
    assert radio_metrics["radio_semantic_accuracy"] == pytest.approx(1.0)
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
        variants=["v1_hierarchical"],
        budgets=[0],
        seeds=[0],
    )
    limited = build_loso_run_plan(
        validation_cfg,
        target_scenes=[34],
        variants=["v0_flat", "v1_hierarchical", "v5_adapter_proto"],
        budgets=[0, 10],
        seeds=[0],
        max_runs=2,
    )
    skipped = build_loso_run_plan(validation_cfg, target_scenes=[34], skip_scenes=[34], variants=["v1_hierarchical"], budgets=[0], seeds=[0])
    single = build_loso_run_plan(
        validation_cfg,
        target_scene=33,
        source_scenes=None,
        variants=["v1_hierarchical"],
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
        "v1_hierarchical",
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


@pytest.mark.parametrize("variant", ["v2_shared_private", "shared_private", "v3_decoupled", "decoupled"])
def test_hist_beam_rejects_retired_knowledge_decoupling_variants(variant: str):
    cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")

    with pytest.raises(ValueError, match="retired"):
        HistBeamFusionNet(variant=variant, feature_size=8, d_model=16, num_classes=16, group_size=4, num_heads=4)

    with pytest.raises(ValueError, match="retired"):
        build_loso_run_plan(cfg, target_scenes=[34], variants=[variant], budgets=[0], seeds=[0])


def test_mmw_loso_plan_uses_availability_claim_guard(tmp_path: Path):
    scenario_a = "Town10_skybridge_seed24"
    scenario_b = "Town10_crossroad_seed24"
    availability_path = tmp_path / "data_availability.json"
    prepared_a = tmp_path / "MMW" / "sunny" / "Prepared" / scenario_a
    prepared_b = tmp_path / "MMW" / "sunny" / "Prepared" / scenario_b

    _write_json(
        availability_path,
        {
            "dataset_family": "MMW",
            "ready_scenario_count": 1,
            "claim_scope": "single_scene_smoke",
            "cross_scene_claim_allowed": False,
            "entries": [
                {
                    "status": "single_scene_ready",
                    "condition": "sunny",
                    "town": "Town10",
                    "scenario": scenario_a,
                    "prepared_root": str(prepared_a),
                    "window_count": 8,
                }
            ],
        },
    )
    smoke_cfg = {
        "data": {"dataset": {"type": "mmw"}},
        "model": {"modalities": ["mmwave"]},
        "loso": {
            "dataset_family": "MMW",
            "data_availability_path": str(availability_path),
            "protocol": "scenario_loso",
        },
    }

    smoke_plan = build_loso_run_plan(
        smoke_cfg,
        variants=["v5_adapter_proto"],
        budgets=[0, 5],
        seeds=[0],
        max_runs=1,
    )

    assert smoke_plan["dataset_family"] == "MMW"
    assert smoke_plan["claim_scope"] == "single_scene_smoke"
    assert smoke_plan["cross_scene_claim_allowed"] is False
    assert smoke_plan["planned_run_count"] == 2
    assert len(smoke_plan["runs"]) == 1
    assert smoke_plan["runs"][0]["fold"] == f"smoke_{scenario_a}"
    assert smoke_plan["runs"][0]["source_scenes"] == [scenario_a]
    assert smoke_plan["runs"][0]["cross_scene_claim_allowed"] is False
    assert smoke_cfg["loso"]["scene_data_roots"][scenario_a] == str(tmp_path / "MMW" / "sunny")
    assert smoke_cfg["loso"]["scene_csv_names"][scenario_a]["test_csv_name"].endswith("splits/test.csv")

    _write_json(
        availability_path,
        {
            "dataset_family": "MMW",
            "ready_scenario_count": 2,
            "claim_scope": "scenario_loso",
            "cross_scene_claim_allowed": True,
            "entries": [
                {
                    "status": "ready_for_loso",
                    "condition": "sunny",
                    "town": "Town10",
                    "scenario": scenario_a,
                    "prepared_root": str(prepared_a),
                    "window_count": 8,
                },
                {
                    "status": "ready_for_loso",
                    "condition": "sunny",
                    "town": "Town10",
                    "scenario": scenario_b,
                    "prepared_root": str(prepared_b),
                    "window_count": 8,
                },
            ],
        },
    )
    loso_cfg = {
        "data": {"dataset": {"type": "mmw"}},
        "model": {"modalities": ["mmwave"]},
        "loso": {"dataset_family": "MMW", "data_availability_path": str(availability_path)},
    }

    loso_plan = build_loso_run_plan(loso_cfg, variants=["v1_hierarchical"], budgets=[0], seeds=[0])

    assert loso_plan["claim_scope"] == "scenario_loso"
    assert loso_plan["cross_scene_claim_allowed"] is True
    assert loso_plan["planned_run_count"] == 2
    assert {run["target_scene"] for run in loso_plan["runs"]} == {scenario_a, scenario_b}
    assert all(run["cross_scene_claim_allowed"] is True for run in loso_plan["runs"])
    assert all(run["target_scene"] not in run["source_scenes"] for run in loso_plan["runs"])

    filtered_plan = build_loso_run_plan(
        loso_cfg,
        target_scene=scenario_a,
        variants=["v4_adapter"],
        budgets=[5],
        seeds=[0],
    )

    assert filtered_plan["planned_run_count"] == 1
    assert [run["target_scene"] for run in filtered_plan["runs"]] == [scenario_a]
    assert filtered_plan["runs"][0]["source_scenes"] == [scenario_b]

    explicit_source_plan = build_loso_run_plan(
        loso_cfg,
        target_scene=scenario_a,
        source_scenes=[scenario_b],
        variants=["v4_adapter"],
        budgets=[5],
        seeds=[0],
    )

    assert explicit_source_plan["runs"][0]["source_scenes"] == [scenario_b]

    _write_mmw_preflight_scene(tmp_path / "MMW" / "sunny", scenario_a)
    _write_mmw_preflight_scene(tmp_path / "MMW" / "sunny", scenario_b)
    preflight = run_loso_execute_preflight(loso_plan, loso_cfg, tmp_path / "out")

    assert preflight["status"] == "passed"
    assert set(preflight["checked_scenes"]) == {scenario_a, scenario_b}

    loso_cfg["loso"]["max_runs"] = 1
    result = run_hist_beam_loso(
        loso_cfg,
        args=_loso_args(tmp_path, execute=True, variants="v5_adapter_proto", budgets="0"),
        stage_executor=FakeStageExecutor(),
    )
    with Path(result["execution"]["summary_paths"]["json"]).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["claim_scope"] == "scenario_loso"
    assert summary["cross_scene_claim_allowed"] is True
    assert summary["runs"][0]["dataset_family"] == "MMW"
    assert summary["runs"][0]["condition"] == "sunny"
    assert summary["runs"][0]["town"] == "Town10"
    assert summary["runs"][0]["prototype_status"] is None or summary["runs"][0]["prototype_status"] == "effective"


def test_mmw_sensor_assisted_config_plan_and_forward_kwargs(tmp_path: Path):
    scenarios = ("Town10_skybridge_seed24", "Town10_crossroad_seed24", "Town10_Hroad_seed42")
    availability_path = tmp_path / "data_availability.json"
    entries = []
    for scenario in scenarios:
        prepared = tmp_path / "MMW" / "sunny" / "Prepared" / scenario
        entries.append(
            {
                "status": "ready_for_loso",
                "condition": "sunny",
                "town": "Town10",
                "scenario": scenario,
                "prepared_root": str(prepared),
                "window_count": 8,
            }
        )
    _write_json(
        availability_path,
        {
            "dataset_family": "MMW",
            "ready_scenario_count": len(entries),
            "claim_scope": "scenario_loso",
            "cross_scene_claim_allowed": True,
            "entries": entries,
        },
    )
    cfg = load_config(ROOT / "configs/hist_beam/mmw_sensor_assisted_quick_validation.yaml")
    cfg["loso"]["data_availability_path"] = str(availability_path)

    enabled = resolve_enabled_modalities(cfg)
    plan = build_loso_run_plan(cfg)
    seq_length = cfg["model"]["seq_length_student"]
    num_pred = cfg["model"]["student"]["num_pred"]
    inputs = prepare_task_inputs(
        {
            "image": torch.randn(2, seq_length, 3, 224, 224),
            "gps": torch.randn(2, seq_length, 3),
            "lidar": torch.randn(2, seq_length, 3, 224, 224),
            "mmwave": torch.randn(2, seq_length, 64),
        },
        "fusion",
        model_cfg=cfg["model"]["student"],
        seq_length=seq_length,
        num_pred=num_pred,
        device=torch.device("cpu"),
    )

    assert cfg["data"]["dataset"]["seq_len"] == 5
    assert cfg["model"]["seq_length_teacher"] == 5
    assert seq_length == 5
    assert cfg["data"]["dataset"]["num_pred"] == 3
    assert cfg["model"]["num_pred"] == 3
    assert num_pred == 3
    assert enabled == ("image", "gps", "lidar")
    assert "radar" not in enabled
    assert "mmwave" not in enabled
    assert "radar_batch" not in inputs
    assert "mmwave_batch" not in inputs
    assert set(inputs) >= {"image_batch", "gps_batch", "lidar_batch"}
    assert inputs["image_batch"].shape[1] == seq_length + num_pred - 1
    assert inputs["gps_batch"].shape[1] == seq_length + num_pred - 1
    assert inputs["lidar_batch"].shape[1] == seq_length + num_pred - 1
    assert plan["profile"] == "sensor_assisted_quick_validation"
    assert plan["matrix"]["budgets"] == [10]
    assert plan["matrix"]["seeds"] == [0, 1]
    assert plan["matrix"]["is_full_budget_seed_sweep"] is False
    assert cfg["data"]["dataloader"]["num_workers"] == 0
    assert cfg["training"]["cpu_threads"]["intra_op"] == 2
    assert cfg["training"]["cpu_threads"]["inter_op"] == 1
    assert {run["budget"] for run in plan["runs"]} == {10}
    assert {run["seed"] for run in plan["runs"]} == {0, 1}
    assert {"v1_hierarchical", "v4_adapter", "v6_radio_proto", "v8_path_proto", "adapter_path_proto", "v6_full_finetune"} <= {
        run["variant"] for run in plan["runs"]
    }
    assert all("radar" not in run["enabled_modalities"] for run in plan["runs"])
    assert all("mmwave" not in run["enabled_modalities"] for run in plan["runs"])


@pytest.mark.parametrize("variant", ["v1_hierarchical", "v4_adapter", "v5_adapter_proto", "v6_radio_proto", "v8_path_proto", "v6_full_finetune"])
def test_hist_beam_sensor_assisted_variants_build(variant: str):
    model = HistBeamFusionNet(
        modalities=["image", "gps", "lidar"],
        feature_size=8,
        d_model=16,
        num_classes=16,
        num_pred=3,
        group_size=4,
        variant=variant,
        num_heads=4,
        num_layers=1,
        image_encoder={"type": "legacy_cnn"},
        radio_semantic={"enabled": variant == "v6_radio_proto", "use_radio_head": variant == "v6_radio_proto"},
        path_semantic={"enabled": variant == "v8_path_proto", "use_path_head": variant == "v8_path_proto"},
        num_radio_classes=12,
        num_path_classes=12,
    )

    assert model.modalities == ("image", "gps", "lidar")
    assert model.num_pred == 3


def test_sensor_assisted_summary_records_deltas_last_beam_and_v8_comparisons(tmp_path: Path):
    base = {
        "fold": "target_scene",
        "target_scene": "Town10_skybridge_seed24",
        "source_scenes": ["Town10_crossroad_seed24"],
        "budget": 10,
        "seed": 0,
        "dataset_family": "MMW",
        "profile": "sensor_assisted_quick_validation",
        "enabled_modalities": ["image", "gps", "lidar"],
        "excluded_sensitive_fields": ["mmwave", "csi", "channel", "path", "beam_power"],
        "matrix_scope": "quick_validation",
        "quick_validation": True,
        "status": "completed",
        "stages": [],
        "artifacts": {},
        "checkpoint_reuse": {},
        "failure_reason": None,
    }

    def record(variant: str, adapted_top1: float | None, trainable_ratio: float | None = 0.1):
        prediction_setup = _strict_mmw_prediction_setup()
        metrics = {
            "source_train": {
                "throughput_config": {
                    "num_workers": 0,
                    "image_cache_policy": "off",
                    "lidar_cache_policy": "auto",
                    "lidar_cache_dir": "lidar_bev_cache",
                    "cpu_threads": {"enabled": True, "intra_op": 2},
                }
            },
            "source_only_target_test_eval": {
                "top1": 0.4,
                "top3": 0.5,
                "top5": 0.6,
                "normalized_received_power": 0.7,
                "beam_power_loss_db": 1.5,
                "prediction_setup": prediction_setup,
                "degradation_baselines": {
                    "last_beam": {"available": True, "top1": [0.9], "top3": [1.0], "avg_top1": 0.9, "avg_top3": 1.0}
                },
            },
        }
        if adapted_top1 is not None:
            metrics["target_adaptation"] = {
                "trainable_ratio": trainable_ratio,
                "adaptation_time_seconds": 2.0,
                "used_target_beam_power_for_training": False,
                "used_target_path_label_for_training": False,
                "used_target_radio_label_for_training": False,
            }
            metrics["adapted_target_test_eval"] = {
                "top1": adapted_top1,
                "top3": adapted_top1 + 0.1,
                "top5": adapted_top1 + 0.2,
                "normalized_received_power": 0.8,
                "beam_power_loss_db": 1.0,
                "path_semantic_accuracy": 0.5 if variant == "v8_path_proto" else None,
                "radio_semantic_accuracy": 0.6 if variant == "v6_radio_proto" else None,
                "prediction_setup": prediction_setup,
                "degradation_baselines": {
                    "last_beam": {"available": True, "top1": [0.9], "top3": [1.0], "avg_top1": 0.9, "avg_top3": 1.0}
                },
            }
        return {"run_id": variant, "variant": variant, "metrics": metrics, **base}

    records = [
        record("v1_hierarchical", None, None),
        record("v6_radio_proto", 0.55),
        record("v8_path_proto", 0.62),
        record("adapter_path_proto", 0.50),
        record("v6_full_finetune", 0.58, 1.0),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])

    summary = json.loads(Path(summary_paths["json"]).read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    v8 = next(row for row in summary["runs"] if row["variant"] == "v8_path_proto")

    assert v8["adapted_source_top1_delta"] == pytest.approx(0.22)
    assert v8["adapted_source_normalized_received_power_delta"] == pytest.approx(0.1)
    assert v8["adapted_source_beam_power_loss_db_delta"] == pytest.approx(-0.5)
    assert v8["negative_transfer"] is False
    assert v8["last_beam_avg_top1"] == pytest.approx(0.9)
    assert v8["last_beam_baseline_type"] == "diagnostic"
    assert v8["last_beam_comparable_baseline"] is False
    assert v8["enabled_modalities"] == ["image", "gps", "lidar"]
    assert v8["lidar_cache_policy"] == "auto"
    assert v8["sensitive_field_usage"]["used_target_beam_power_for_training"] is False
    assert any(item["comparison"] == "v6_radio_vs_v8_path" and item["status"] == "complete" for item in conclusion["comparisons"])
    assert any(item["comparison"] == "path_condition_off_vs_on" and item["status"] == "complete" for item in conclusion["comparisons"])
    assert any(item["comparison"] == "v8_path_vs_full_finetune" and item["status"] == "complete" for item in conclusion["comparisons"])


def test_quick_validation_excludes_ineligible_candidate_from_win_loss(tmp_path: Path):
    records = [
        _quick_conclusion_record("v1_hierarchical", source_top1=0.4),
        _quick_conclusion_record(
            "v4_adapter",
            source_top1=0.4,
            adapted_top1=0.8,
            adaptation_metrics={
                "main_conclusion_eligible": False,
                "eligibility_reasons": ["target_radio_label_supervision"],
                "used_target_radio_label_for_training": True,
            },
        ),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))

    comparison = next(
        item
        for item in conclusion["comparisons"]
        if item["comparison"] == "adapter_vs_source_only" and item["candidate_variant"] == "v4_adapter"
    )
    assert comparison["status"] == "inconclusive"
    assert "candidate_better_than_source_only" not in comparison
    assert comparison["missing"][0]["variant"] == "v4_adapter"
    assert comparison["missing"][0]["reason"] == "run_excluded_from_main_conclusion"
    assert "target_radio_label_supervision" in comparison["missing"][0]["eligibility_reasons"]
    assert conclusion["eligible_run_count"] == 1
    assert conclusion["excluded_run_count"] == 1
    assert conclusion["exclusion_reason_histogram"]["target_radio_label_supervision"] == 1
    assert conclusion["excluded_runs"][0]["variant"] == "v4_adapter"


def test_loso_summary_keeps_legacy_kd_baseline_supplemental(tmp_path: Path):
    records = [
        _quick_conclusion_record("v1_hierarchical", source_top1=0.4),
        _quick_conclusion_record(
            "v4_adapter",
            source_top1=0.4,
            adapted_top1=0.9,
            adaptation_metrics={
                "distillation_enabled": True,
                "method_family": "legacy_kd",
                "distillation_type": "logits_kd",
                "teacher_checkpoint": "outputs/scene31/fusion_teacher_no_kd/checkpoints/best.pth",
                "teacher_source": "registry",
                "distillation_lifecycle": "legacy_kd",
                "baseline_role": "optional_baseline",
                "reproduction_scope": "historical_reproduction",
                "main_conclusion_eligible": False,
            },
        ),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])
    summary = json.loads(Path(summary_paths["json"]).read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))

    mainline = next(row for row in summary["runs"] if row["variant"] == "v1_hierarchical")
    legacy = next(row for row in summary["runs"] if row["variant"] == "v4_adapter")
    comparison = next(
        item
        for item in conclusion["comparisons"]
        if item["comparison"] == "adapter_vs_source_only" and item["candidate_variant"] == "v4_adapter"
    )

    assert mainline["method_family"] != "legacy_kd"
    assert mainline["distillation_enabled"] is False
    assert mainline["main_conclusion_eligible"] is True
    assert legacy["method_family"] == "legacy_kd"
    assert legacy["distillation_enabled"] is True
    assert legacy["distillation_type"] == "logits_kd"
    assert legacy["teacher_checkpoint"].endswith("best.pth")
    assert legacy["teacher_source"] == "registry"
    assert legacy["distillation_lifecycle"] == "legacy_kd"
    assert legacy["baseline_role"] == "optional_baseline"
    assert legacy["reproduction_scope"] == "historical_reproduction"
    assert legacy["main_conclusion_eligible"] is False
    assert "legacy_kd_supplemental" in legacy["eligibility_reasons"]
    assert summary["eligible_run_count"] == 1
    assert summary["excluded_run_count"] == 1
    assert summary["exclusion_reason_histogram"]["legacy_kd_supplemental"] == 1
    assert conclusion["excluded_run_count"] == 1
    assert conclusion["exclusion_reason_histogram"]["legacy_kd_supplemental"] == 1
    assert conclusion["excluded_runs"][0]["variant"] == "v4_adapter"
    assert comparison["status"] == "inconclusive"
    assert comparison["missing"][0]["reason"] == "run_excluded_from_main_conclusion"
    assert "legacy_kd_supplemental" in comparison["missing"][0]["eligibility_reasons"]


def test_quick_validation_excludes_ineligible_mmw_split_from_main_conclusion(tmp_path: Path):
    ineligible_setup = _strict_mmw_prediction_setup(
        split_strategy="group_safe_time_block",
        strict=False,
        eligibility_reasons=["guard_band_violation"],
    )
    records = [
        _quick_conclusion_record("v1_hierarchical", source_top1=0.4, prediction_setup=ineligible_setup),
        _quick_conclusion_record("v4_adapter", source_top1=0.4, adapted_top1=0.8),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])
    summary = json.loads(Path(summary_paths["json"]).read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))

    baseline = next(row for row in summary["runs"] if row["variant"] == "v1_hierarchical")
    assert baseline["main_conclusion_eligible"] is False
    assert baseline["split_strategy"] == "group_safe_time_block"
    assert "guard_band_violation" in baseline["eligibility_reasons"]
    assert conclusion["excluded_run_count"] == 1
    assert conclusion["exclusion_reason_histogram"]["guard_band_violation"] == 1


def test_quick_validation_excludes_unknown_mmw_split_from_main_conclusion(tmp_path: Path):
    records = [
        _quick_conclusion_record("v1_hierarchical", source_top1=0.4, include_prediction_setup=False),
        _quick_conclusion_record("v4_adapter", source_top1=0.4, adapted_top1=0.8),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    summary = json.loads(Path(summary_paths["json"]).read_text(encoding="utf-8"))

    baseline = next(row for row in summary["runs"] if row["variant"] == "v1_hierarchical")
    assert baseline["main_conclusion_eligible"] is False
    assert baseline["split_eligibility"] == "unknown"
    assert "split_eligibility_unknown" in baseline["eligibility_reasons"]
    assert summary["excluded_run_count"] == 1
    assert summary["exclusion_reason_histogram"]["split_eligibility_unknown"] == 1


def test_quick_validation_marks_excluded_baseline_comparison_inconclusive(tmp_path: Path):
    records = [
        _quick_conclusion_record(
            "v1_hierarchical",
            source_top1=0.4,
            adaptation_metrics={
                "main_conclusion_eligible": False,
                "eligibility_reasons": ["target_leakage"],
                "target_leakage": True,
            },
        ),
        _quick_conclusion_record("v4_adapter", source_top1=0.4, adapted_top1=0.8),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))

    comparison = next(
        item
        for item in conclusion["comparisons"]
        if item["comparison"] == "adapter_vs_source_only" and item["candidate_variant"] == "v4_adapter"
    )
    assert comparison["status"] == "inconclusive"
    assert comparison["missing"][0]["variant"] == "v1_hierarchical"
    assert comparison["missing"][0]["reason"] == "run_excluded_from_main_conclusion"
    assert "target_leakage" in comparison["missing"][0]["eligibility_reasons"]
    assert conclusion["inconclusive_comparison_count"] >= 1


def test_quick_validation_treats_prototype_no_op_as_ineligible_evidence(tmp_path: Path):
    records = [
        _quick_conclusion_record("v1_hierarchical", source_top1=0.4),
        _quick_conclusion_record(
            "v5_adapter_proto",
            source_top1=0.4,
            adapted_top1=0.9,
            adaptation_metrics={
                "prototype_status": "no_op",
                "prototype_coverage": 0.0,
                "prototype_loss_mean": None,
                "proto_type": "radio_semantic",
            },
        ),
        _quick_conclusion_record("v6_full_finetune", source_top1=0.4, adapted_top1=0.6),
    ]

    summary_paths = write_loso_execute_summary(tmp_path, records, status="completed")
    conclusion_path = write_quick_validation_conclusion(tmp_path, records, summary_paths["json"])
    summary = json.loads(Path(summary_paths["json"]).read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))

    proto_row = next(row for row in summary["runs"] if row["variant"] == "v5_adapter_proto")
    assert proto_row["main_conclusion_eligible"] is False
    assert "prototype_no_op" in proto_row["eligibility_reasons"]
    comparison = next(item for item in conclusion["comparisons"] if item["comparison"] == "adapter_proto_vs_full_finetune")
    assert comparison["status"] == "inconclusive"
    assert "adapter_proto_better_than_full_finetune" not in comparison
    assert comparison["missing"][0]["variant"] == "v5_adapter_proto"
    assert "prototype_no_op" in comparison["missing"][0]["eligibility_reasons"]


def test_loso_stage_local_helpers_delay_target_dataset_construction(monkeypatch):
    calls: list[tuple[str, str]] = []

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, split: str, scene: str):
            self.split = split
            self.scene_id = scene
            self.scene_slug = scene
            self.enabled_modalities = ("image", "gps", "mmwave")
            self.root_csv = None
            self.gps_scaler = object()
            self.mmwave_scaler = object()

        def __len__(self):
            return 2

        def __getitem__(self, index):  # noqa: ARG002
            return {"input_beam": torch.tensor([0]), "target_beam": torch.tensor([0])}

    def fake_build_dataset(cfg, split, **kwargs):  # noqa: ANN001, ARG001
        scene = str(cfg["data"]["dataset"].get("scene"))
        calls.append((scene, split))
        return TinyDataset(split, scene)

    monkeypatch.setattr(loso_data_module, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(
        loso_data_module,
        "_split_target_dataset_records",
        lambda dataset, dataset_cfg, adapt_fraction, seed: (  # noqa: ARG005
            loso_data_module.TargetSplit(adapt_indices=(0,), test_indices=(1,), metadata={}),
            (0, 1),
        ),
    )
    cfg = {
        "data": {"dataset": {"type": "mmw"}, "dataloader": {"batch_size": 1, "num_workers": 0}},
        "model": {"student": {"modalities": ["image", "gps", "mmwave"]}},
        "experiment": {"task": "fusion"},
        "hist_beam": {"source_sampling": {"scene_balance": {"enabled": True}}},
    }
    fold = {"dataset_family": "MMW", "target_scene": "target", "source_scenes": ["source_a", "source_b"]}

    source = build_loso_source_train_loader(cfg, fold)

    assert set(source) >= {"source_train", "normalization_kwargs", "source_sampling"}
    assert source["source_sampling"]["scene_balance_enabled"] is True
    assert source["source_sampling"]["strategy"] == "scene_balanced_weighted_sampler"
    assert type(source["source_train"].sampler).__name__ == "WeightedRandomSampler"
    assert calls == [("source_a", "train"), ("source_b", "train")]

    calls.clear()
    target = build_loso_target_stage_loader(
        cfg,
        fold,
        stage="target_adapt",
        dataset_kwargs=source["normalization_kwargs"],
    )

    assert "target_adapt" in target
    assert calls == [("target", "test")]


def test_prototype_decision_skips_source_only_and_generates_for_proto_variants():
    cfg = {"hist_beam": {"prototype": {"strategy": "auto"}}}

    skipped = _prototype_decision({"variant": "v0_flat"}, cfg, source_variant="v0_flat")
    generated = _prototype_decision({"variant": "v5_adapter_proto"}, cfg, source_variant="v1_hierarchical")

    assert skipped["generate"] is False
    assert skipped["status"] == "skipped"
    assert "source_only_variant" in skipped["reason"]
    assert generated["generate"] is True
    assert "variant_requires_prototype" in generated["reason"]


@pytest.mark.parametrize(
    ("variant", "expected_source"),
    [
        ("v4_adapter", "v1_hierarchical"),
        ("v5_adapter_proto", "v1_hierarchical"),
        ("v6_radio_proto", "v1_hierarchical"),
        ("adapter_radio_proto", "v1_hierarchical"),
        ("v8_path_proto", "v1_hierarchical"),
        ("adapter_path_proto", "v1_hierarchical"),
        ("v7_shared_physical_private_residual", "v1_hierarchical"),
        ("v8_target_prior_head", "v1_hierarchical"),
        ("v9_input_conditioned_target_adaptation", "v1_hierarchical"),
        ("image_source_only", "v0_flat"),
        ("image_v8_target_prior_head", "v0_flat"),
        ("image_v9_sector_proto", "v0_flat"),
    ],
)
def test_retained_routes_do_not_fall_back_to_retired_source_variant(variant: str, expected_source: str):
    assert _source_variant_for({"variant": variant}) == expected_source


def test_hist_beam_stage_cfg_defaults_to_no_kd_lineage(tmp_path: Path):
    cfg = load_config(ROOT / "configs/hist_beam/quick_smoke.yaml")
    run = {
        "fold": "target_scene34",
        "target_scene": 34,
        "source_scenes": [31, 32, 33],
        "variant": "v5_adapter_proto",
        "budget": 10,
        "seed": 0,
    }

    for stage_name in ("source_train", "source_only_target_test_eval", "target_adaptation", "adapted_target_test_eval"):
        stage_cfg = _stage_cfg(
            cfg,
            run,
            variant="v5_adapter_proto",
            stage_name=stage_name,
            stage_dir=tmp_path / stage_name,
        )
        distillation = stage_cfg["distillation"]

        assert distillation["type"] == "no_kd"
        assert distillation["teacher_model_name"] is None
        assert distillation["method_family"] == "mainline_no_kd"
        assert distillation["lifecycle"] == "active_mainline_no_kd"
        assert distillation["main_conclusion_eligible"] is True


def test_hist_beam_loso_execute_uses_runner_and_records_stage_metadata(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    executor = MetadataAssertingStageExecutor()
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v1_hierarchical", budgets="0"),
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
    plan = build_loso_run_plan(cfg, target_scenes=[34], variants=["v1_hierarchical"], budgets=[0], seeds=[0])

    preflight = run_loso_execute_preflight(plan, cfg, tmp_path / "out")

    assert preflight["status"] == "failed"
    assert any(error["resource_type"] == "data_root" and error["scene"] == 31 for error in preflight["errors"])
    assert any("missing" in str(error["path"]) for error in preflight["errors"])


def test_mmw_loso_preflight_reports_public_radar_preparation_command(tmp_path: Path):
    scene = "Town10_skybridge_seed24"
    condition_root = tmp_path / "MMW" / "sunny"
    _write_mmw_preflight_scene_without_radar_columns(condition_root, scene)
    cfg = {
        "experiment": {"task": "fusion"},
        "data": {
            "dataset": {
                "type": "mmw",
                "condition": "sunny",
                "scene": scene,
                "data_root": str(condition_root),
                "train_csv_name": f"Prepared/{scene}/splits/train.csv",
                "test_csv_name": f"Prepared/{scene}/splits/test.csv",
                "seq_len": 1,
                "num_pred": 1,
                "enabled_modalities": ["radar"],
            }
        },
        "model": {"modalities": ["radar"], "student": {"modalities": ["radar"]}},
        "loso": {
            "scene_data_roots": {scene: str(condition_root)},
            "scene_csv_names": {
                scene: {
                    "train_csv_name": f"Prepared/{scene}/splits/train.csv",
                    "test_csv_name": f"Prepared/{scene}/splits/test.csv",
                }
            },
        },
    }
    plan = {
        "runs": [
            {
                "fold": "target_scene",
                "target_scene": scene,
                "source_scenes": [],
                "variant": "v1_hierarchical",
                "budget": 0,
                "seed": 0,
                "enabled_modalities": ["radar"],
            }
        ]
    }

    preflight = run_loso_execute_preflight(plan, cfg, tmp_path / "out")

    assert preflight["status"] == "failed"
    radar_errors = [error for error in preflight["errors"] if error["resource_type"] == "radar_derived_csv"]
    assert radar_errors
    message = radar_errors[0]["message"]
    assert "kd-sensing-preprocess" in message
    assert "configs/preprocess/mmw_radar_maps.yaml" in message
    assert scene in message
    assert str(condition_root) in message
    assert "train_with_radar.csv" in message or "test_with_radar.csv" in message


def test_mmw_loso_preflight_does_not_import_dataset_private_materializers():
    source = (SRC / "kd_sensing" / "engine" / "hist_beam_loso_execution.py").read_text(encoding="utf-8")
    preflight = (SRC / "kd_sensing" / "engine" / "hist_beam_loso_preflight.py").read_text(encoding="utf-8")

    assert "from kd_sensing.data.datasets.mmw import _ensure" not in source
    assert "from kd_sensing.data.datasets.mmw import _ensure" not in preflight
    assert "kd_sensing.preprocessing.mmw_radar" in preflight


def test_hist_beam_loso_execute_smoke_writes_summary_and_conclusion(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v1_hierarchical,v5_adapter_proto,v6_full_finetune", budgets="0"),
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
    assert summary["completed_count"] == 3
    assert summary["failed_count"] == 0
    assert summary["missing_count"] == 0
    assert "eligible_run_count" in summary
    assert "excluded_run_count" in summary
    assert "exclusion_reason_histogram" in summary
    assert any(row["variant"] == "v5_adapter_proto" for row in summary["runs"])
    proto_run = next(run for run in execution["runs"] if run["variant"] == "v5_adapter_proto")
    assert proto_run["checkpoint_reuse"]["source_train"]["enabled"] is True
    assert Path(proto_run["artifacts"]["run_metadata_path"]).exists()
    run_metadata = json.loads(Path(proto_run["artifacts"]["run_metadata_path"]).read_text(encoding="utf-8"))
    assert run_metadata["checkpoint_reuse"]["source_train"]["enabled"] is True
    assert run_metadata["stages"][-1]["name"] == "summary"
    proto_row = next(row for row in summary["runs"] if row["variant"] == "v5_adapter_proto")
    assert "sensitive_field_usage" in proto_row
    assert "main_conclusion_eligible" in proto_row
    assert "eligibility_reasons" in proto_row
    assert conclusion_payload["summary_path"] == str(summary_json)
    assert conclusion_payload["source_paths"]["summary_path"] == str(summary_json)
    assert "eligible_run_count" in conclusion_payload
    assert "inconclusive_comparison_count" in conclusion_payload


def test_hist_beam_loso_execute_keyboard_interrupt_writes_partial_summary(tmp_path: Path):
    cfg = _loso_fixture_config(tmp_path)
    result = run_hist_beam_loso(
        cfg,
        args=_loso_args(tmp_path, execute=True, variants="v1_hierarchical,v5_adapter_proto", budgets="0"),
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
            "v1_hierarchical",
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


def _quick_conclusion_record(
    variant: str,
    *,
    source_top1: float,
    adapted_top1: float | None = None,
    adaptation_metrics: dict | None = None,
    prediction_setup: dict | None = None,
    include_prediction_setup: bool = True,
):
    setup = prediction_setup or _strict_mmw_prediction_setup()
    metrics = {
        "source_only_target_test_eval": {
            "top1": source_top1,
            "top3": source_top1 + 0.1,
            "top5": source_top1 + 0.2,
            "coarse_accuracy": source_top1 + 0.05,
            "fine_offset_accuracy": source_top1 - 0.05,
        }
    }
    if include_prediction_setup:
        metrics["source_only_target_test_eval"]["prediction_setup"] = setup
    if adapted_top1 is not None:
        metrics["adapted_target_test_eval"] = {
            "top1": adapted_top1,
            "top3": adapted_top1 + 0.1,
            "top5": adapted_top1 + 0.2,
            "coarse_accuracy": adapted_top1 + 0.05,
            "fine_offset_accuracy": adapted_top1 - 0.05,
            "radio_semantic_accuracy": 0.6 if variant == "v6_radio_proto" else None,
            "path_semantic_accuracy": 0.5 if variant == "v8_path_proto" else None,
        }
        if include_prediction_setup:
            metrics["adapted_target_test_eval"]["prediction_setup"] = setup
    if adaptation_metrics is not None or variant not in SOURCE_ONLY_VARIANTS:
        base_adaptation = {
            "trainable_ratio": 0.1 if variant != "v6_full_finetune" else 1.0,
            "adaptation_time_seconds": 1.0,
            "used_target_beam_for_training": False,
            "used_target_beam_power_for_training": False,
            "used_target_csi_for_training": False,
            "used_target_path_params_for_training": False,
            "used_target_path_descriptor_for_training": False,
            "used_target_path_label_for_training": False,
            "used_target_radio_label_for_training": False,
            "sensitive_field_policy": {"allow_target_sensitive_supervision_in_main_conclusion": False},
        }
        if adaptation_metrics:
            base_adaptation.update(adaptation_metrics)
        metrics["target_adaptation"] = base_adaptation
    return {
        "run_id": variant,
        "fold": "target_scene",
        "target_scene": "Town10_skybridge_seed24",
        "source_scenes": ["Town10_crossroad_seed24"],
        "variant": variant,
        "budget": 10,
        "seed": 0,
        "dataset_family": "MMW",
        "profile": "sensor_assisted_quick_validation",
        "enabled_modalities": ["image", "gps", "lidar"],
        "excluded_sensitive_fields": ["mmwave", "csi", "channel", "path", "beam_power"],
        "matrix_scope": "quick_validation",
        "quick_validation": True,
        "status": "completed",
        "stages": [],
        "artifacts": {},
        "failure_reason": None,
        "metrics": metrics,
    }


def _strict_mmw_prediction_setup(
    *,
    split_strategy: str = "group_safe_time_block",
    strict: bool = True,
    eligibility_reasons: list[str] | None = None,
) -> dict:
    reasons = list(eligibility_reasons or [])
    split_payload = {
        "csv_path": "dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p6_group_safe/test.csv",
        "csv_name": "test.csv",
        "num_samples": 8,
        "split_protocol": "mmw_sequence_split_v2",
        "split_strategy": split_strategy,
        "split_protocol_version": "mmw_sequence_split_v2",
        "strict_validation_eligible": strict,
        "eligibility_reasons": reasons,
        "leakage_diagnostics": {
            "train_test_frame_overlap_count": 0,
            "guard_band_violations": 0,
        },
        "split_seed": 42,
        "split_sequence_count": 8,
        "split_num_samples": 8,
        "split_metadata_path": "dataset/MMW/sunny/Prepared/Town10_skybridge_seed24/splits/l5p6_group_safe/split_metadata.json",
    }
    return {
        "split_protocol": split_payload["split_protocol"],
        "split_strategy": split_payload["split_strategy"],
        "split_protocol_version": split_payload["split_protocol_version"],
        "strict_validation_eligible": strict,
        "eligibility_reasons": reasons,
        "leakage_diagnostics": split_payload["leakage_diagnostics"],
        "split_metadata_path": split_payload["split_metadata_path"],
        "split_seed": split_payload["split_seed"],
        "split_sequence_count": split_payload["split_sequence_count"],
        "split_num_samples": split_payload["split_num_samples"],
        "splits": {"test": split_payload},
    }


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


def _loso_args(tmp_path: Path, *, execute: bool, variants: str = "v1_hierarchical", budgets: str = "0"):
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


def _radio_adaptation_cfg(*, allow_radio: bool) -> dict:
    return {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {
            "seq_length_student": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {
                "type": "hist_beam_fusion",
                "modalities": ["gps"],
                "num_classes": 16,
                "group_size": 4,
            },
        },
        "hist_beam": {
            "group_size": 4,
            "loss_weights": {"radio_semantic": 1.0},
            "adaptation": {
                "entropy_weight": 0.01,
                "prototype_weight": 0.0,
                "allow_labeled_target_radio_supervision": allow_radio,
            },
        },
    }


def _path_adaptation_cfg(*, allow_path: bool) -> dict:
    return {
        "experiment": {"task": "fusion"},
        "data": {"dataset": {"seq_len": 1, "num_pred": 1}},
        "model": {
            "seq_length_student": 1,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {
                "type": "hist_beam_fusion",
                "modalities": ["gps"],
                "num_classes": 16,
                "group_size": 4,
            },
        },
        "hist_beam": {
            "group_size": 4,
            "loss_weights": {"lambda_path": 0.3, "lambda_path_reg": 0.0},
            "adaptation": {
                "entropy_weight": 0.01,
                "prototype_weight": 0.0,
                "allow_labeled_target_path_supervision": allow_path,
            },
        },
    }


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


def _write_mmw_preflight_scene(condition_root: Path, scenario: str) -> None:
    root = condition_root / "Prepared" / scenario
    power_path = root / "beam_power" / "cav_1" / "000000.txt"
    _write_power_vector(power_path, label=3)
    rel_power_path = power_path.relative_to(condition_root)
    split_root = root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    headers = ["seq_index", "beam1", "future_beam1", "mmwave1"]
    row = {
        "seq_index": "0",
        "beam1": str(rel_power_path),
        "future_beam1": str(rel_power_path),
        "mmwave1": str(rel_power_path),
    }
    for csv_name in ("train.csv", "test.csv"):
        with (split_root / csv_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)


def _write_mmw_preflight_scene_without_radar_columns(condition_root: Path, scenario: str) -> None:
    root = condition_root / "Prepared" / scenario
    power_path = root / "beam_power" / "cav_1" / "000000.txt"
    _write_power_vector(power_path, label=3)
    rel_power_path = power_path.relative_to(condition_root)
    split_root = root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    headers = ["seq_index", "beam1", "future_beam1", "mmwave1"]
    row = {
        "seq_index": "0",
        "beam1": str(rel_power_path),
        "future_beam1": str(rel_power_path),
        "mmwave1": str(rel_power_path),
    }
    for csv_name in ("train.csv", "test.csv"):
        with (split_root / csv_name).open("w", encoding="utf-8", newline="") as f:
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
