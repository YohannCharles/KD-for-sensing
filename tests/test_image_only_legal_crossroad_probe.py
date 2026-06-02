from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.cli.hist_beam_loso import build_loso_run_plan  # noqa: E402
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.hist_beam_adaptation import apply_hist_beam_adaptation_strategy  # noqa: E402
from kd_sensing.engine.hist_beam_image_only import (  # noqa: E402
    expected_feature_cache_metadata,
    filter_image_only_batch,
    load_image_feature_cache,
    validate_feature_cache_metadata,
    write_image_feature_cache,
)
from kd_sensing.engine.hist_beam_loso_config import _stage_cfg  # noqa: E402
from kd_sensing.engine.hist_beam_loso_summary import row_eligibility  # noqa: E402
from kd_sensing.engine.hist_beam_losses import compute_hist_beam_loss  # noqa: E402
from kd_sensing.evaluation.hist_beam_outputs import collapse_diagnostics_payload, write_confusion_by_true_beam, write_prediction_histogram  # noqa: E402
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402


def _image_only_cfg() -> dict:
    return {
        "experiment": {"task": "fusion", "seed": 0},
        "data": {"dataset": {"enabled_modalities": ["image"], "seq_len": 2, "num_pred": 1}},
        "model": {
            "modalities": ["image"],
            "seq_length_student": 2,
            "num_pred": 1,
            "downsample_ratio": 1,
            "student": {"modalities": ["image"], "num_classes": 8, "image_only": {"fusion_mode": "identity"}},
        },
        "hist_beam": {
            "protocol": {"image_only": True},
            "modalities": ["image"],
            "disabled_modalities": ["gps", "lidar", "radar", "mmwave", "csi"],
            "excluded_sensitive_fields": ["gps", "lidar", "radar", "mmwave", "csi", "channel", "path", "beam_power"],
        },
    }


def test_image_only_batch_allowlist_records_available_vs_consumed_fields():
    batch = {
        "image": torch.randn(2, 2, 3, 224, 224),
        "target_beam": torch.tensor([[1], [2]]),
        "gps": torch.randn(2, 2, 3),
        "beam_power": torch.randn(2, 1, 8),
        "metadata": {
            "sample_id": ["a", "b"],
            "split": ["target_adapt", "target_adapt"],
            "beam_power_path": ["raw/a.txt", "raw/b.txt"],
        },
    }

    filtered = filter_image_only_batch(batch, _image_only_cfg(), stage="target_adaptation")

    assert set(filtered) == {"image", "target_beam", "metadata"}
    assert "gps" not in filtered
    assert "beam_power" not in filtered
    assert "beam_power" in filtered["metadata"]["image_only_available_fields"][0]
    assert filtered["metadata"]["image_only_consumed_input_fields"][0] == ["target_support.image"]
    assert filtered["metadata"]["image_only_consumed_label_fields"][0] == ["target_support.target_beam"]


def test_image_only_dataloader_one_batch_drops_disabled_fields():
    samples = [
        {
            "image": torch.randn(2, 3, 224, 224),
            "target_beam": torch.tensor([1]),
            "gps": torch.randn(2, 3),
            "beam_power": torch.randn(1, 8),
            "metadata": {"sample_id": f"s{idx}", "split": "target_adapt", "beam_power_path": f"{idx}.txt"},
        }
        for idx in range(2)
    ]
    batch = next(iter(DataLoader(samples, batch_size=2)))
    filtered = filter_image_only_batch(batch, _image_only_cfg(), stage="target_adaptation")

    assert "image" in filtered
    assert "target_beam" in filtered
    assert "gps" not in filtered
    assert "beam_power" not in filtered


def test_image_only_model_forward_and_linear_probe_freeze_policy():
    model = HistBeamFusionNet(
        modalities=["image"],
        feature_size=8,
        d_model=16,
        num_classes=8,
        num_pred=1,
        group_size=2,
        variant="image_only_v8_v9_probe",
        num_heads=4,
        num_layers=1,
        image_encoder={"type": "legacy_cnn"},
        image_only={"fusion_mode": "identity"},
        v8={"mode": "target_linear_probe", "use_adapter": False, "use_target_prior": False, "beta_prior": 0.0},
    )

    output = model(image_batch=torch.randn(2, 2, 3, 224, 224))
    strategy = apply_hist_beam_adaptation_strategy(model, "image_target_linear_probe")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert output["logits"].shape == (2, 1, 8)
    assert output["logits_final"].shape == (2, 1, 8)
    assert output["target_logits"].shape == (2, 1, 8)
    assert output["features"].shape == (2, 1, 16)
    assert output["hist_beam"]["image_only_fusion_mode"] == "identity"
    assert strategy["image_target_linear_probe_freeze_strategy"] is True
    assert trainable
    assert all(name.startswith("target_head") for name in trainable)


def test_image_only_i1_i2_i3_forward_loss_backward_smoke():
    modes = [
        ("image_target_linear_probe", "v8_target_prior_head", {"mode": "target_linear_probe", "use_adapter": False, "use_target_prior": False, "beta_prior": 0.0}, None),
        ("image_v8_target_prior_head", "v8_target_prior_head", {"mode": "target_prior_head", "use_adapter": True, "use_target_prior": True, "learnable_beta_prior": True}, None),
        ("image_v9_sector_proto", "v9_input_conditioned_target_adaptation", {"mode": "target_prior_head", "use_adapter": True, "use_target_prior": True}, {"prototype_type": "sector", "sector_size": 2}),
    ]
    labels = torch.tensor([[1], [2]])
    for probe_mode, variant, v8_cfg, v9_cfg in modes:
        model = HistBeamFusionNet(
            modalities=["image"],
            feature_size=8,
            d_model=16,
            num_classes=8,
            num_pred=1,
            group_size=2,
            variant=variant,
            num_heads=4,
            num_layers=1,
            image_encoder={"type": "legacy_cnn"},
            image_only={"fusion_mode": "identity"},
            v8=v8_cfg,
            v9=v9_cfg,
        )
        if probe_mode == "image_v8_target_prior_head":
            model.set_target_prior_from_labels([1, 2], sigma=1.0)
        if probe_mode == "image_v9_sector_proto":
            model.set_target_prior_from_labels([1, 2], sigma=1.0)
            model.set_target_prototypes_from_features(torch.randn(2, 16), labels[:, 0], prototype_type="sector", sector_size=2)
        strategy = "image_target_linear_probe" if probe_mode == "image_target_linear_probe" else (
            "v9_target_head_only" if probe_mode == "image_v9_sector_proto" else "v8_target_head_only"
        )
        apply_hist_beam_adaptation_strategy(model, strategy)
        output = model(image_batch=torch.randn(2, 2, 3, 224, 224))
        loss = compute_hist_beam_loss(
            output,
            labels,
            cfg={"hist_beam": {"variant": variant, "num_classes": 8, "group_size": 2, "v8": v8_cfg, "v9": v9_cfg or {}}},
            num_classes=8,
        )
        loss.total.backward()
        assert loss.total.isfinite()
        assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_prediction_histogram_and_confusion_include_image_only_collapse_fields(tmp_path: Path):
    labels = torch.tensor([[1], [2], [2], [4]])
    outputs = torch.zeros(4, 1, 8)
    outputs[:, :, 2] = 3.0

    hist_path = write_prediction_histogram(tmp_path / "prediction_hist.json", labels, outputs, num_classes=8)
    confusion_path = write_confusion_by_true_beam(tmp_path / "confusion_by_true_beam.json", labels, outputs, num_classes=8)
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    confusion = json.loads(confusion_path.read_text(encoding="utf-8"))

    assert hist["unique_pred_beams"] == 1
    assert hist["top1_pred_beam_ratio"] == 1.0
    assert hist["top5_pred_beam_ratio"] == 1.0
    assert hist["within_1_acc"] == 0.75
    assert confusion["confusion_by_true_beam"]["2"]["2"] == 2

    collapse = collapse_diagnostics_payload(
        labels,
        outputs,
        num_classes=8,
        target_logits=outputs - 1.0,
        target_prior_bias=torch.zeros_like(outputs),
        prototype_logits=torch.ones_like(outputs),
    )
    assert "[v9-sector] top predicted beams before proto" in collapse
    assert "[v9-sector] top predicted beams after proto" in collapse


def test_image_only_eligibility_uses_consumed_fields_not_raw_available_fields():
    row = {
        "dataset_family": "MMW",
        "town": "Town10",
        "target_scene": "Town10_crossroad_seed24",
        "run_status": "completed",
        "strict_validation_eligible": True,
        "available_fields": ["image", "target_beam", "beam_power", "path"],
        "consumed_fields": {
            "target_adaptation": {
                "consumed_input_fields": ["target_support.image"],
                "consumed_label_fields": ["target_support.target_beam"],
            },
            "target_test": {
                "consumed_input_fields": ["target_test.image"],
                "consumed_label_fields": ["target_test.target_beam:evaluation_only"],
            },
        },
    }
    legal = row_eligibility(row, {"main_conclusion_eligible": True, "consumed_fields": row["consumed_fields"]}, {})
    illegal_consumed = {
        "target_adaptation": {
            "consumed_input_fields": ["target_support.image", "target_support.beam_power"],
            "consumed_label_fields": ["target_test.target_beam"],
        }
    }
    illegal = row_eligibility(row, {"main_conclusion_eligible": True, "consumed_fields": illegal_consumed}, {})

    assert legal["main_conclusion_eligible"] is True
    assert illegal["main_conclusion_eligible"] is False
    assert "target_oracle_consumed:target_adaptation:target_support.beam_power" in illegal["eligibility_reasons"]
    assert "target_oracle_consumed:target_adaptation:target_test.target_beam" in illegal["eligibility_reasons"]


def test_image_only_config_plan_and_stage_defaults(tmp_path: Path):
    cfg = load_config(ROOT / "configs/hist_beam/image_only_legal_crossroad_probe.yaml")
    plan = build_loso_run_plan(
        cfg,
        variants=list(cfg["loso"]["variants"]),
        budgets=list(cfg["loso"]["budgets"]),
        seeds=list(cfg["loso"]["seeds"]),
    )
    run_by_variant = {run["variant"]: run for run in plan["runs"]}
    stage_cfg = _stage_cfg(
        cfg,
        run_by_variant["image_v9_sector_proto"],
        variant="image_v9_sector_proto",
        stage_name="target_adaptation",
        stage_dir=tmp_path,
    )
    linear_stage_cfg = _stage_cfg(
        cfg,
        run_by_variant["image_target_linear_probe"],
        variant="image_target_linear_probe",
        stage_name="target_adaptation",
        stage_dir=tmp_path,
    )

    assert plan["enabled_modalities"] == ["image"]
    assert set(run_by_variant) == {
        "image_source_only",
        "image_target_linear_probe",
        "image_v8_target_prior_head",
        "image_v9_sector_proto",
    }
    assert run_by_variant["image_source_only"]["stages"] == ["source_train", "source_only_target_test_eval", "summary"]
    assert stage_cfg["model"]["student"]["variant"] == "v9_input_conditioned_target_adaptation"
    assert stage_cfg["model"]["student"]["modalities"] == ["image"]
    assert stage_cfg["hist_beam"]["v9"]["prototype_type"] == "sector"
    assert stage_cfg["hist_beam"]["v9"]["use_beam_proto"] is False
    assert linear_stage_cfg["hist_beam"]["adaptation"]["strategy"] == "image_target_linear_probe"


def test_image_feature_cache_read_write_metadata_and_target_test_label_scope(tmp_path: Path):
    expected = expected_feature_cache_metadata(
        _image_only_cfg(),
        {"source_scenes": ["A"], "target_scene": "B", "budget": 10},
        checkpoint="ckpt.pth",
        feature_dim=16,
    )
    cache = write_image_feature_cache(
        tmp_path / "target_test.pt",
        features=torch.randn(2, 16),
        labels=torch.tensor([[1], [2]]),
        metadata_rows=[{"scene_slug": "B", "sample_id": "a", "split": "target_test"}, {"scene_slug": "B", "sample_id": "b", "split": "target_test"}],
        split="target_test",
        cache_metadata=expected,
    )
    eval_payload = load_image_feature_cache(
        cache["path"],
        expected_metadata=expected,
        split="target_test",
        scope="evaluation",
    )
    adapt_payload = load_image_feature_cache(
        cache["path"],
        expected_metadata=expected,
        split="target_test",
        scope="adaptation",
    )

    assert eval_payload["labels"].shape == (2, 1)
    assert adapt_payload["labels"] is None
    assert adapt_payload["labels_unavailable_reason"] == "target_test_labels_blocked_for_adaptation_scope"
    validate_feature_cache_metadata(dict(expected), expected)
    actual = dict(expected)
    actual["feature_dim"] = 32

    try:
        validate_feature_cache_metadata(actual, expected)
    except ValueError as exc:
        assert "feature_dim" in str(exc)
    else:
        raise AssertionError("feature cache metadata mismatch should fail")
