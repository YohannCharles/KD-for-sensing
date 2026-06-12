from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.config import load_config
from kd_sensing.data.transform_ops.gps import load_relative_xy_sequence
from kd_sensing.engine.batch import prepare_fusion_inputs, prepare_gps_bev_xy_inputs
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.run_metadata import prediction_setup_metadata
from kd_sensing.evaluation.bev_fusion_2604_report import (
    bev_fusion_2604_model_size,
    build_bev_fusion_2604_report,
)
from kd_sensing.registries import MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/fusion/experiments/bev_fusion_2604"


def test_gps_bev_xy_loader_keeps_unstandardized_relative_xy(tmp_path: Path):
    gps_dir = tmp_path / "gps"
    gps_dir.mkdir()
    bs_dir = tmp_path / "bs"
    bs_dir.mkdir()
    gps_paths = []
    bs_paths = []
    for idx in range(3):
        gps_path = gps_dir / f"ue_{idx}.txt"
        bs_path = bs_dir / f"bs_{idx}.txt"
        gps_path.write_text(f"{33.4194132 + idx * 1e-5}\n{-111.9288816 + idx * 1e-5}\n", encoding="utf-8")
        bs_path.write_text("33.41932083333333\n-111.92902222222223\n", encoding="utf-8")
        gps_paths.append(str(gps_path.relative_to(tmp_path)))
        bs_paths.append(str(bs_path.relative_to(tmp_path)))

    xy = load_relative_xy_sequence(tmp_path, gps_paths, bs_paths, seq_len=3)

    assert xy.shape == (3, 2)
    assert np.isfinite(xy).all()
    assert not np.allclose(xy.mean(axis=0), np.zeros(2), atol=1e-3)


def test_prepare_gps_bev_xy_inputs_shape_and_padding():
    batch = {"gps_bev_xy": torch.ones(2, 5, 2)}

    xy = prepare_gps_bev_xy_inputs(batch, seq_length=3, num_pred=2, device=torch.device("cpu"))

    assert xy.shape == (2, 4, 2)
    assert torch.all(xy[:, -1, :] == 0)


def test_bev_fusion_2604_config_family_loads_contracts():
    full = load_config(CONFIG_ROOT / "paper_full.yaml")
    smoke = load_config(CONFIG_ROOT / "smoke.yaml")
    ablation = load_config(CONFIG_ROOT / "ablations/gps_global_only.yaml")

    assert full["data"]["dataset"]["type"] == "deepsense6g"
    assert full["data"]["dataset"]["train_scenes"] == [32, 33, 34]
    assert full["data"]["dataset"]["seq_len"] == 5
    assert full["data"]["dataset"]["num_pred"] == 1
    assert full["model"]["primary"]["type"] == "bev_fusion_2604"
    assert full["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar"]
    assert full["model"]["primary"]["bev_size"] == [128, 128]
    assert full["model"]["primary"]["d_model"] == 256
    assert full["model"]["primary"]["temporal_core"]["num_layers"] == 4
    assert full["loss"]["type"] == "focal_loss"
    assert full["loss"]["gamma"] == 2
    assert full["training"]["optimizer"]["type"] == "adamw"
    assert full["training"]["lr"] == pytest.approx(1e-4)
    assert full["training"]["weight_decay"] == pytest.approx(1e-2)
    assert full["evaluation"]["metric_profile"] == "2604_linear_topk"
    assert full["evaluation"]["dba_distance_mode"] == "linear"
    assert smoke["data"]["dataset"]["type"] == "synthetic_sequence"
    assert smoke["data"]["dataset"]["mock_data"] is True
    assert smoke["model"]["primary"]["paper_approximation"] is True
    assert ablation["model"]["primary"]["gps_pathway"] == "global_only"
    assert ablation["model"]["primary"]["ablation_name"] == "gps_global_only"


def test_bev_fusion_2604_forward_smoke_and_gps_pathway_ablation():
    import_default_components()
    cfg = load_config(CONFIG_ROOT / "smoke.yaml")
    model = MODELS.build(cfg["model"]["primary"])
    batch = _synthetic_batch()

    output = adapt_model_output(model(**batch))
    size = bev_fusion_2604_model_size(model)

    assert tuple(output.logits.shape) == (2, 1, 64)
    assert size["total_params"] > 0
    assert size["trainable_params"] > 0
    assert output.input_features is not None
    assert output.output_features is not None
    assert output.diagnostics["bev_feature_shape"] == tuple(output.diagnostics["bev_features"].shape)
    assert output.diagnostics["gps_pathway"] == "dual_path"
    assert output.diagnostics["effective_modalities"] == ("image", "radar", "gps", "lidar")

    missing_xy = dict(batch)
    missing_xy.pop("gps_bev_xy_batch")
    with pytest.raises(ValueError, match="gps_bev_xy_batch"):
        model(**missing_xy)

    global_cfg = deepcopy(cfg["model"]["primary"])
    global_cfg["gps_pathway"] = "global_only"
    global_model = MODELS.build(global_cfg)
    global_output = adapt_model_output(global_model(**missing_xy))
    assert tuple(global_output.logits.shape) == (2, 1, 64)
    assert global_output.diagnostics["gps_pathway"] == "global_only"


def test_bev_fusion_2604_prepare_fusion_inputs_passes_gps_bev_xy():
    cfg = load_config(CONFIG_ROOT / "smoke.yaml")
    raw = {
        "image": torch.rand(2, 5, 3, 64, 64),
        "radar_ra": torch.rand(2, 5, 128, 64),
        "radar_da": torch.rand(2, 5, 128, 64),
        "gps": torch.rand(2, 5, 3),
        "gps_bev_xy": torch.rand(2, 5, 2),
        "lidar": torch.rand(2, 5, 3, 16, 16),
    }

    inputs = prepare_fusion_inputs(
        raw,
        seq_length=5,
        num_pred=1,
        device=torch.device("cpu"),
        modalities=cfg["model"]["primary"]["modalities"],
        image_profile=cfg["model"]["primary"].get("image_profile"),
        input_profiles=cfg["model"]["primary"].get("input_profiles"),
    )

    assert sorted(inputs) == ["gps_batch", "gps_bev_xy_batch", "image_batch", "lidar_batch", "radar_batch"]
    assert inputs["gps_bev_xy_batch"].shape == (2, 5, 2)


def test_bev_fusion_2604_report_and_runtime_metadata():
    cfg = load_config(CONFIG_ROOT / "smoke.yaml")
    setup = prediction_setup_metadata(cfg)

    assert setup["primary_model"] == "bev_fusion_2604"
    assert setup["paper_exact_split_available"] is False
    assert setup["mock_data"] is True
    assert setup["paper_approximation"] is True
    assert setup["bev_shape"] == [8, 8]
    assert setup["gps_pathway"] == "dual_path"

    report = build_bev_fusion_2604_report(
        {
            "S32": {"DBA": 0.8, "top1": 0.7, "sample_count": 10},
            "S33": {"DBA": 0.9, "top1": 0.8, "sample_count": 30},
        },
        split_protocol="synthetic",
        seed=7,
        mock_data=True,
        paper_approximation=True,
    )

    assert report["metric_profile"] == "2604_linear_topk"
    assert report["dba_distance_mode"] == "linear"
    assert report["macro_linear_dba"] == pytest.approx(0.85)
    assert report["weighted_overall_linear_dba"] == pytest.approx(0.875)
    assert report["scene_breakdown"]["S32"]["paper_target_linear_dba"] == pytest.approx(0.866)
    assert report["mock_data"] is True
    assert "mock_or_synthetic" in report["caveat"]


def _synthetic_batch() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.rand(2, 5, 3, 64, 64),
        "radar_batch": torch.rand(2, 5, 2, 128, 64),
        "gps_batch": torch.rand(2, 5, 3),
        "gps_bev_xy_batch": torch.zeros(2, 5, 2),
        "lidar_batch": torch.rand(2, 5, 3, 16, 16),
    }
