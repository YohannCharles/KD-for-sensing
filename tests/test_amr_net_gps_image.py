from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kd_sensing.baselines.amr_net_gps_image.metrics import paper_aligned_metric_summary
from kd_sensing.baselines.amr_net_gps_image.preset import (
    AMR_NET_GPS_IMAGE_DISPLAY_NAME,
    build_model_group_config,
    paper_model_groups,
)
from kd_sensing.baselines.amr_net_gps_image.report import run_amr_net_gps_image
from kd_sensing.baselines.amr_net_gps_image.source_audit import (
    build_default_source_audit,
    ensure_claim_status_allowed,
    missing_official_requirements,
)
from kd_sensing.config import load_config
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.registries import MODELS, import_default_components
from kd_sensing.utils.runtime_output_layout import runtime_output_scope_from_config


ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / "configs/baselines/amr_net_gps_image.yaml"


def test_amr_net_gps_image_source_audit_records_conflict_and_gates_official_claim():
    audit = build_default_source_audit()
    payload = audit.to_dict()

    assert payload["article_number"] == "11282996"
    assert payload["doi"] == "10.1109/JIOT.2025.3641184"
    assert payload["article_metadata_conflict"] is True
    assert payload["local_substitute"]["article_number"] == "10000718"
    assert payload["local_substitute"]["dataset_scene"] == 23
    assert payload["local_substitute"]["author_code_commit"] == "4b80592ed3517726f3fc5af441db25acd1811d3e"
    assert len(payload["digest"]) == 64

    missing = missing_official_requirements(payload)
    assert "article_metadata_conflict" in missing
    assert "official_weights" in missing
    with pytest.raises(ValueError, match="official_reproduction is blocked"):
        ensure_claim_status_allowed("official_reproduction", payload)
    assert ensure_claim_status_allowed("mock_smoke", payload) == "mock_smoke"


def test_scenario23_descriptor_and_paper_preset_load_with_lidar_named_csv():
    scene = resolve_deepsense_scene("ScEnArIo23")
    assert scene.scene_id == 23
    assert scene.scene_slug == "scene23"
    assert scene.default_data_root == "dataset/DeepSense6G/scenario23"

    cfg = load_config(PRESET)
    assert cfg["experiment"]["name"] == AMR_NET_GPS_IMAGE_DISPLAY_NAME
    dataset = cfg["data"]["dataset"]
    assert dataset["scene_id"] == 23
    assert dataset["scene_slug"] == "scene23"
    assert dataset["data_root"] == "dataset/DeepSense6G/scenario23"
    assert dataset["train_csv_name"] == "train_seqs_RA_GPS_LIDAR.csv"
    assert dataset["test_csv_name"] == "test_seqs_RA_GPS_LIDAR.csv"
    assert dataset["use_lidar"] is False
    assert cfg["model"]["modalities"] == ["image", "gps"]
    assert cfg["model"]["primary"]["modalities"] == ["image", "gps"]

    scope = runtime_output_scope_from_config(cfg)
    assert scope is not None
    assert scope.slug == "scene23"

    overridden = load_config(
        PRESET,
        [
            "data.dataset.data_root=/tmp/custom_s23",
            "data.dataset.train_csv_name=my_train.csv",
            "data.dataset.test_csv_name=my_test.csv",
        ],
    )
    assert overridden["data"]["dataset"]["data_root"] == "/tmp/custom_s23"
    assert overridden["data"]["dataset"]["train_csv_name"] == "my_train.csv"
    assert overridden["data"]["dataset"]["test_csv_name"] == "my_test.csv"
    assert overridden["data"]["dataset"]["scene_slug"] == "scene23"

    with pytest.raises(ValueError, match="Unsupported DeepSense6G scene"):
        resolve_deepsense_scene("scenario999")


@pytest.mark.parametrize(
    "override",
    [
        "data.dataset.use_lidar=true",
        "model.modalities=[image,gps,lidar]",
        "model.primary.checkpoint_path=outputs/analysis/deepsense6g_gps_lidar_bgam/best.pth",
    ],
)
def test_amr_net_gps_image_preset_rejects_lidar_and_bgam_overrides(override: str):
    with pytest.raises(ValueError, match="only allows image and GPS"):
        load_config(PRESET, [override])


def test_amr_net_gps_image_model_groups_forward_and_metadata_smoke():
    import_default_components()
    batch_size = 2
    labels = torch.tensor([[0], [3]])
    for group in paper_model_groups():
        model = MODELS.build(build_model_group_config(group.group_id, smoke=True))
        if group.group_id == "image_only":
            raw = model(image_batch=torch.rand(batch_size, 1, 3, 64, 64))
        elif group.group_id == "gps_only":
            raw = model(gps_batch=torch.rand(batch_size, 1, 2))
        else:
            raw = model(
                image_batch=torch.rand(batch_size, 1, 3, 64, 64),
                gps_batch=torch.rand(batch_size, 1, 2),
            )
        output = adapt_model_output(raw)
        assert output.logits.shape == (batch_size, 1, 64)
        metadata = model.training_strategy_metadata()
        assert metadata["model_name"] == AMR_NET_GPS_IMAGE_DISPLAY_NAME
        assert metadata["enabled_modalities"] == list(group.enabled_modalities)
        assert metadata["claim_status"] == group.claim_status
        assert metadata["uses_lidar"] is False
        summary = paper_aligned_metric_summary(
            output.logits,
            labels,
            model_group=group.group_id,
            claim_status=group.claim_status,
            mock_data=True,
        )
        assert set(["top1", "top3", "top5", "DBA", "metric_profile"]) <= set(summary)
        assert summary["mock_data"] is True


def test_amr_net_gps_image_report_writer_marks_mock_and_blocked(tmp_path: Path):
    report = run_amr_net_gps_image(output_dir=tmp_path, mock=True)

    assert report["mock_data"] is True
    assert report["model_name"] == AMR_NET_GPS_IMAGE_DISPLAY_NAME
    assert report["claim_status"] == "mock_smoke"
    assert report["source_audit"]["article_metadata_conflict"] is True
    assert report["checkpoint_provenance"]["mock_data"] is True
    assert report["scenario"]["scene_slug"] == "scene23"
    assert report["dataset"]["use_lidar"] is False
    assert {row["model_group"] for row in report["metrics"]} == {"image_only", "gps_only", "image_gps_fusion"}
    assert (tmp_path / "source_audit.json").exists()
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "reproduction_manifest.json").exists()
    assert (tmp_path / "report.md").exists()
