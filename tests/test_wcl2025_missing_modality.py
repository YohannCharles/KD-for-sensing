import json
from pathlib import Path

import torch

from kd_sensing.baselines.rmbp_mm import (
    LOCAL_SUBSTITUTE_CONFIG,
    apply_missing_modality_condition,
    build_condition_summary,
    build_local_substitute_model_config,
    build_source_audit_manifest,
    run_source_audit_dry_run,
    write_condition_summary,
)
from kd_sensing.config import load_config
from kd_sensing.data.difficulty import DifficultyContext, apply_configured_difficulty
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.registries import MODELS, import_default_components


ROOT = Path(__file__).resolve().parents[1]


def test_source_audit_manifest_blocks_official_and_selects_local_substitute(tmp_path: Path):
    manifest = build_source_audit_manifest(
        output_root=tmp_path,
        generated_at="2026-06-22T00:00:00Z",
        command_args=("unit-test",),
    )

    assert manifest["schema_version"].endswith(".v1")
    assert manifest["paper"]["doi"] == "10.1109/LWC.2025.3591611"
    assert manifest["source_audit"]["code"]["availability"] == "unavailable"
    assert manifest["branches"]["official_code"]["status"] == "blocked"
    assert manifest["branches"]["local_substitute"]["claim_status"] == "local_substitute"
    assert manifest["claim_status"] == "local_substitute"
    assert manifest["branches"]["local_substitute"]["config"] == LOCAL_SUBSTITUTE_CONFIG
    assert {"code", "checkpoint", "dataset", "modalities", "split", "metric_profile", "training_recipe"} <= set(
        manifest["source_audit"]
    )

    written = run_source_audit_dry_run(output_root=tmp_path, generated_at="2026-06-22T00:00:00Z")
    path = Path(written["metadata"]["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path
    assert payload["branches"]["official_code"]["claim_status"] == "blocked"


def test_condition_summary_marks_strict_mismatch_out_of_ranking(tmp_path: Path):
    strict = {
        "split": "S32-S34_train_S31-S34_test",
        "scene_set": [31, 32, 33, 34],
        "label_space": "beam64",
        "metric_profile": "wcl2025_missing_modality_local_topk_dba",
        "sample_count": 10,
        "seed": 17,
        "difficulty_digest": "digest-a",
    }
    summary = build_condition_summary(
        [
            {
                "condition_id": "clean",
                "affected_modalities": [],
                "top1": 0.4,
                "top5": 0.8,
                "dba": 0.7,
                **strict,
            },
            {
                "condition_id": "missing_gps",
                "affected_modalities": ["gps"],
                "top1": 0.2,
                "top5": 0.5,
                "dba": 0.4,
                **strict,
            },
            {
                "condition_id": "missing_image_lidar",
                "affected_modalities": ["image", "lidar"],
                "top1": 0.1,
                "top5": 0.3,
                "dba": 0.2,
                **{**strict, "seed": 99},
            },
        ],
        strict_protocol=strict,
    )

    assert summary["summary"]["condition_types"] == [
        "clean",
        "multi_modality_missing",
        "single_modality_missing",
    ]
    assert len(summary["strict_ranking_rows"]) == 2
    mismatch = summary["conditions"][2]
    assert mismatch["claim_status"] == "not_comparable"
    assert mismatch["eligible_for_strict_ranking"] is False
    assert mismatch["strict_mismatches"][0]["field"] == "seed"

    paths = write_condition_summary(summary, output_root=tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).read_text(encoding="utf-8").splitlines()[0].startswith("condition_id,")


def test_local_substitute_config_loads_and_declares_wcl_metadata():
    cfg = load_config(ROOT / LOCAL_SUBSTITUTE_CONFIG)
    primary = cfg["model"]["primary"]
    profile = cfg["difficulty"]["profiles"][0]

    assert primary["type"] == "modular_sequence"
    assert primary["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert primary["representation_core"]["type"] == "token_transformer"
    assert primary["paper_metadata"]["model_group"] == "RMBP-MM"
    assert primary["paper_metadata"]["baseline_scope"] == "local_experimental_baseline"
    assert "not paper-specific imputation/channel attention" in primary["paper_metadata"]["notes"][-1]
    assert profile["operators"][0]["type"] == "modality_dropout"
    assert profile["operators"][0]["params"]["rates"]["mmwave"] == 0.3
    assert cfg["comparability"]["official_reproduction_status"] == "not_applicable"
    assert cfg["output"]["dir"].startswith("outputs/analysis/local_baselines/rmbp_mm")


def test_wcl_style_modality_dropout_supports_mmwave_inputs():
    cfg = {
        "difficulty": {
            "profiles": [
                {
                    "id": "wcl_mmwave_dropout",
                    "stage": "train",
                    "split": "train",
                    "severity": 1.0,
                    "affected_modalities": ["mmwave"],
                    "operators": [
                        {
                            "type": "modality_dropout",
                            "affected_modalities": ["mmwave"],
                            "rates": {"mmwave": 1.0},
                        }
                    ],
                }
            ]
        }
    }
    batch = {"mmwave": torch.ones(2, 3, 64)}

    result = apply_configured_difficulty(
        batch,
        cfg,
        DifficultyContext(stage="train", split="train", seed=17, epoch=0, step=0),
    ).batch

    assert torch.count_nonzero(result["mmwave"]).item() == 0
    assert result["mmwave_valid_mask"].shape == (2, 3)
    assert result["missing_modality_metadata"]["rates"]["mmwave"] == 1.0


def test_local_substitute_model_config_builds_and_handles_missing_condition():
    import_default_components()
    model_cfg = build_local_substitute_model_config(
        modalities=("radar", "gps", "lidar", "mmwave"),
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )
    model = MODELS.build(model_cfg)
    batch = {
        "radar_batch": torch.randn(2, 1, 2, 128, 64),
        "gps_batch": torch.randn(2, 1, 3),
        "lidar_batch": torch.randn(2, 1, 3, 16, 16),
        "mmwave_batch": torch.randn(2, 1, 64),
    }
    missing = apply_missing_modality_condition(batch, ["gps", "lidar"])

    assert torch.count_nonzero(missing["gps_batch"]).item() == 0
    assert torch.count_nonzero(missing["lidar_batch"]).item() == 0
    assert missing["gps_valid_mask"].shape == (2, 1)

    output = model(**missing)
    adapted = adapt_model_output(output)
    metadata = model.training_strategy_metadata()
    assert adapted.logits.shape == (2, 1, 5)
    assert metadata["model_group"] == "RMBP-MM"
    assert metadata["missing_modality_strategy"] == "zero_imputation_with_modality_dropout_training"
    assert metadata["fusion_type"] == "token_transformer"
    assert "does not implement the paper-specific imputation and channel-attention modules" in metadata["deviation"]
