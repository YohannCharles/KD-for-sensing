import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.diagnostics.jepa_benchmark_manifest import load_benchmark_manifest  # noqa: E402
from kd_sensing.engine.artifacts import final_config_with_runtime  # noqa: E402


def test_config_load_pipeline_characterization_covers_sources_and_overrides():
    entity = load_config(ROOT / "configs/gps/lightweight.yaml")
    virtual_fusion = load_config(ROOT / "configs/fusion/gps_mmwave_lightweight.yaml")
    snapshot = load_config(ROOT / "configs/fusion/all_modalities_snapshot_next_frame_supervised.yaml")
    u_mask_jepa = load_config(ROOT / "configs/fusion/u_mask_beam_jepa_smoke.yaml")
    overridden = load_config(
        ROOT / "configs/fusion/gps_mmwave_lightweight.yaml",
        [
            "experiment.objective=beam",
            "training.early_stopping_metric=val_loss",
            "training.early_stopping_mode=min",
            "data.dataset.scene=32",
        ],
    )

    assert entity["data"]["dataset"]["scene_slug"] == "scene31"
    assert entity["training"]["early_stopping_metric"] == "val_adba"
    assert virtual_fusion["model"]["modalities"] == ["gps", "mmwave"]
    assert virtual_fusion["model"]["primary"]["modalities"] == ["gps", "mmwave"]
    assert "distillation" not in virtual_fusion
    assert snapshot["experiment"]["variant"] == "snapshot_next_frame"
    assert snapshot["experiment"]["uses_history_window"] is False
    assert u_mask_jepa["model"]["primary"]["modalities"] == ["image", "radar", "gps", "lidar"]
    assert overridden["data"]["dataset"]["scene_slug"] == "scene32"
    assert overridden["training"]["early_stopping_metric"] == "val_loss"
    assert overridden["training"]["early_stopping_mode"] == "min"


def test_rbma_missing_workflow_configs_load_without_retired_kd_or_vision():
    base = ROOT / "configs/fusion/experiments/rbma_missing_workflow"
    main = load_config(base / "no_jepa_rbma_proto_kd.yaml")
    baseline = load_config(base / "amber_style_mask_baseline.yaml")

    assert main["model"]["primary"]["fusion_type"] == "reliability_biased_missing_attention"
    assert main["model"]["primary"]["use_jepa_loss"] is False
    assert main["training"]["mask_sampler"] == "pattern_balanced"
    assert main["training"]["use_beam_prototype_alignment"] is True
    assert main["training"]["use_full_to_partial_kd"] is True
    assert main["training"]["kd_teacher_mode"] == "online_full"
    assert main["training"]["epochs"] == 40
    assert main["training"]["patience"] == 20
    assert main["training"]["validation"]["interval_epochs"] == 10
    assert main["data"]["dataset"]["portion"] == 1.0
    assert main["data"]["dataset"]["sample_cache"]["enabled"] is True
    assert main["data"]["dataset"]["sample_cache"]["backend"] == "lmdb"
    assert main["data"]["dataset"]["sample_cache"]["readahead"] is False
    assert main["data"]["dataloader"]["train_batch_size"] == 128
    assert main["data"]["dataloader"]["train_num_workers"] == 2
    assert main["data"]["dataloader"]["train_prefetch_factor"] == 2
    assert main["data"]["dataloader"]["train_persistent_workers"] is False
    assert main["training"]["cpu_threads"] == {"intra_op": 12, "inter_op": 1}
    assert main["output"]["progress"]["enabled"] is True
    assert baseline["model"]["primary"]["fusion_type"] == "weighted_sum"
    assert baseline["training"].get("use_beam_prototype_alignment", False) is False
    assert baseline["training"].get("use_full_to_partial_kd", False) is False

    text = "\n".join(path.read_text(encoding="utf-8") for path in base.glob("*.yaml"))
    assert "vision" not in text
    assert "logits_kd" not in text
    assert "rkd" not in text


def test_predictive_jepa_hybrid_config_loads_and_preserves_existing_jepa_baselines(tmp_path: Path):
    predictive = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_jepa_predictive_hybrid_beambench_fair_lowmem.yaml"
    )
    gps_query = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml"
    )
    resnet_image_gps = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "image_gps_supervised_beambench_fair_lowmem.yaml"
    )

    image_encoder = predictive["model"]["primary"]["encoders"]["image"]
    assert predictive["experiment"]["model_group"] == "jepa_predictive_hybrid"
    assert predictive["data"]["dataset"]["seq_len"] == 4
    assert image_encoder["pooling"] == "hybrid_residual_query"
    assert image_encoder["pooler"]["type"] == "hybrid_residual_query"
    assert image_encoder["pooler"]["residual_alpha_init"] == 0.1
    assert image_encoder["temporal_auxiliary"]["enabled"] is True
    assert predictive["model"]["primary"]["representation_core"]["type"] == "feature_consistency_gate"
    assert predictive["difficulty"]["profiles"][0]["condition"] == "P4_joint_predictive_recovery"
    assert predictive["difficulty"]["profiles"][0]["operators"][0]["affected_modalities"] == ["image", "gps"]
    assert predictive["output"]["dir"].startswith("outputs/analysis/predictive_jepa_robustness")

    metadata = final_config_with_runtime(predictive, run_dir=tmp_path / "predictive")["runtime"]["jepa_downstream"]
    assert metadata["pooler_type"] == "hybrid_residual_query"
    assert metadata["hybrid_residual_query_enabled"] is True
    assert metadata["hybrid_content_queries"] == 2
    assert metadata["hybrid_gps_queries"] == 2
    assert metadata["temporal_auxiliary_enabled"] is True
    assert metadata["representation_core_type"] == "feature_consistency_gate"
    assert metadata["representation_core"]["history_window"] == 3

    assert gps_query["model"]["primary"]["encoders"]["image"]["pooling"] == "gps_query_attention"
    assert resnet_image_gps["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"


def test_tinyvit_image_encoder_override_is_opt_in_and_default_stays_resnet18():
    default_image = load_config(ROOT / "configs/image/supervised.yaml")
    tinyvit_image = load_config(
        ROOT / "configs/image/supervised.yaml",
        ["model.primary.encoders.image.type=tinyvit_5m_scratch_rgb"],
    )
    tinyvit_fusion = load_config(
        ROOT / "configs/fusion/image_gps_supervised.yaml",
        ["model.primary.encoders.image.type=tinyvit_11m_22k_rgb"],
    )

    assert default_image["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert tinyvit_image["model"]["primary"]["encoders"]["image"]["type"] == "tinyvit_5m_scratch_rgb"
    assert tinyvit_fusion["model"]["primary"]["encoders"]["image"]["type"] == "tinyvit_11m_22k_rgb"
    assert tinyvit_image["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert tinyvit_fusion["data"]["dataset"]["image_profile"] == "rgb_imagenet"


def test_legacy_model_registry_surface_root_configs_are_modular():
    expected = {
        "configs/radar/strong.yaml": ("radar", "radar_cnn", "single_gru"),
        "configs/radar/lightweight.yaml": ("radar", "radar_cnn", "single_gru"),
        "configs/radar/supervised.yaml": ("radar", "radar_cnn", "single_gru"),
        "configs/gps/strong.yaml": ("gps", "gps_mlp", "single_gru"),
        "configs/gps/lightweight.yaml": ("gps", "gps_mlp", "single_gru"),
        "configs/gps/supervised.yaml": ("gps", "gps_mlp", "single_gru"),
        "configs/gps/ablation_relative_polar.yaml": ("gps", "gps_mlp", "single_gru"),
        "configs/mmwave/strong.yaml": ("mmwave", "mmwave_mlp", "single_gru"),
        "configs/mmwave/lightweight.yaml": ("mmwave", "mmwave_mlp", "single_gru"),
        "configs/mmwave/supervised.yaml": ("mmwave", "mmwave_mlp", "single_gru"),
    }

    for rel_path, (modality, encoder_type, core_type) in expected.items():
        cfg = load_config(ROOT / rel_path)
        primary = cfg["model"]["primary"]
        assert primary["type"] == "modular_sequence"
        assert primary["modalities"] == [modality]
        assert primary["encoders"][modality]["type"] == encoder_type
        assert primary["representation_core"]["type"] == core_type
        assert primary["heads"]["beam"]["type"] == "beam_head"

    fusion = load_config(ROOT / "configs/fusion/radar_gps_supervised.yaml")
    primary = fusion["model"]["primary"]
    assert primary["type"] == "modular_sequence"
    assert primary["modalities"] == ["radar", "gps"]
    assert primary["encoders"]["radar"]["type"] == "radar_cnn"
    assert primary["encoders"]["gps"]["type"] == "gps_mlp"
    assert primary["representation_core"]["type"] == "early_concat_gru"
    assert primary["heads"]["beam"]["type"] == "beam_head"


def test_amr_net_current_config_loads_without_retired_token():
    cfg = load_config(ROOT / "configs/fusion/amr_net_supervised.yaml")
    primary = cfg["model"]["primary"]
    serialized = repr(cfg)

    assert primary["type"] == "amr_net"
    assert cfg["model"]["modalities"] == ["image", "gps", "lidar"]
    assert cfg["loss"]["amr"]["enabled"] is True
    assert "amr_net_gps_image" not in serialized


def test_geometry_prior_beam_fusion_configs_and_strict_manifest_load():
    fusion = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "geometry_prior_logit_fusion_2604_s32_s34_lowmem.yaml"
    )
    dba = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "geometry_prior_dba_aware_loss_2604_s32_s34_lowmem.yaml"
    )
    teacher = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "geometry_prior_teacher_guided_2604_s32_s34_lowmem.yaml"
    )
    curriculum = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "geometry_prior_mixed_curriculum_2604_s32_s34_lowmem.yaml"
    )
    rerank = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/"
        "safe_residual_rerank_strict_candidate_2604_s32_s34_lowmem.yaml"
    )
    manifest = load_benchmark_manifest(
        ROOT / "configs/diagnostics/geometry_prior_beam_fusion_strict.yaml",
        validate_paths=False,
    )
    rerank_manifest = load_benchmark_manifest(
        ROOT / "configs/diagnostics/real_perturbation_residual_rerank_fusion_strict.yaml",
        validate_paths=False,
    )

    prior_cfg = fusion["model"]["primary"]["geometry_prior"]
    assert fusion["experiment"]["seed"] == 17
    assert fusion["model"]["gps_input_seq_len"] == 2
    assert prior_cfg["enabled"] is True
    assert prior_cfg["type"] == "gps_geometry_prior"
    assert prior_cfg["label_space"] == "beam64"
    assert fusion["model"]["primary"]["logit_fusion"]["type"] == "geometry_prior_logit_fusion"
    assert dba["loss"]["dba_aware"]["enabled"] is True
    assert "distillation" not in dba["loss"]
    assert teacher["loss"]["teacher_guidance"]["mode"] == "opt_in_stabilization"
    assert "teacher_checkpoint" not in teacher["loss"]["teacher_guidance"]
    assert curriculum["training"]["curriculum"]["mode"] == "clean_first_geometry_prior"
    assert rerank["model"]["primary"]["reranker"]["type"] == "safe_residual_beam_reranker"
    assert rerank["model"]["primary"]["reranker"]["max_residual_scale"] == 0.35
    assert rerank["loss"]["safe_rerank"]["enabled"] is True
    assert rerank["loss"]["safe_rerank"]["no_regret_weight"] == 0.25
    assert manifest["comparison_protocol"]["history_window"] == 5
    assert manifest["comparison_protocol"]["gps_input_source_window"] == 2
    assert manifest["comparison_protocol"]["prediction_horizon"] == 1
    assert manifest["comparison_protocol"]["scene_set"] == [32, 33, 34]
    assert manifest["comparison_protocol"]["seed"] == 17
    assert manifest["geometry_prior_claim_gate"]["clean_regression_threshold_dba"] == 0.02
    assert "geometry_prior_logit_fusion" in manifest["models"]
    assert rerank_manifest["evaluation"]["mode"] == "real_forward"
    assert rerank_manifest["geometry_prior_claim_gate"]["require_real_forward_perturbations"] is True
    assert rerank_manifest["models"]["safe_residual_rerank_strict_candidate"]["group"] == "safe_residual_beam_rerank_fusion"


def test_base_overlay_configs_preserve_key_load_semantics():
    cases = [
        (
            "configs/csi/hardening_matrix/A0_clean_full_strong.yaml",
            "configs/csi/hardening_matrix/_base/csi_only.yaml",
            "A0_clean_full_strong",
        ),
        (
            "configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml",
            "configs/fusion/csi_hardening_matrix/_base/gps_csi.yaml",
            "E1_gps_clean_csi_joint",
        ),
    ]

    for overlay_path, base_path, overlay_id in cases:
        overlay = load_config(ROOT / overlay_path)
        base = load_config(ROOT / base_path)

        assert _config_signature(overlay) == _config_signature(base)
        assert overlay["config_resolution"]["style"] == "base+overlay"
        assert overlay["config_resolution"]["overlay_id"] == overlay_id


def test_physics_informed_mmw_configs_load_and_preserve_boundaries():
    debug = load_config(ROOT / "configs/fusion/physics_informed_mmw_debug.yaml")
    no_physics = load_config(ROOT / "configs/fusion/physics_informed_mmw_no_physics.yaml")
    full = load_config(ROOT / "configs/fusion/physics_informed_mmw_full_multimodal.yaml")
    vision = load_config(ROOT / "configs/fusion/physics_informed_mmw_vision_only.yaml")
    partial = load_config(ROOT / "configs/fusion/physics_informed_mmw_partial_csi_multimodal.yaml")
    history = load_config(ROOT / "configs/fusion/physics_informed_mmw_history_csi_multimodal.yaml")
    paper_debug = load_config(ROOT / "configs/fusion/physics_informed_mmw_paper_debug.yaml")
    oracle = load_config(ROOT / "configs/fusion/physics_informed_mmw_oracle_full_csi.yaml")

    assert debug["data"]["dataset"]["type"] == "mmw"
    assert debug["model"]["primary"]["type"] == "pinn_multimodal_beam"
    assert debug["loss"]["type"] == "cross_entropy"
    assert debug["loss"]["physics"]["enabled"] is True
    assert debug["data"]["use_csi_input"] is False
    assert debug["data"]["csi_input_mode"] == "none"
    assert no_physics["loss"]["physics"]["enabled"] is False
    assert full["model"]["primary"]["modalities"] == ["image", "gps", "lidar", "mmwave"]
    assert vision["model"]["primary"]["modalities"] == ["image"]
    assert vision["data"]["csi_input_mode"] == "none"
    assert partial["model"]["primary"]["modalities"] == ["image", "csi"]
    assert partial["data"]["csi_input_mode"] == "partial"
    assert history["model"]["primary"]["modalities"] == ["image", "csi"]
    assert history["data"]["csi_input_mode"] == "history"
    assert paper_debug["model"]["primary"]["frontend"]["type"] == "paper_modal_tokenizers"
    assert paper_debug["model"]["primary"]["frontend"]["formal_experiment_eligible"] is False
    assert not paper_debug["model"]["primary"]["frontend"]["encoders"]["image"].get("checkpoint_path")
    assert oracle["model"]["primary"]["modalities"] == ["csi"]
    assert oracle["data"]["csi_input_mode"] == "oracle_full"
    assert oracle["data"]["allow_oracle_full_csi_input"] is True
    assert "scripts/inspect_dataset.py" not in repr(debug)


def _config_signature(cfg: dict) -> dict:
    dataset = cfg["data"]["dataset"]
    model = cfg["model"]
    primary = model["primary"]
    training = cfg["training"]
    return {
        "experiment_name": cfg["experiment"]["name"],
        "task": cfg["experiment"]["task"],
        "objective": cfg["experiment"].get("objective"),
        "dataset_type": dataset["type"],
        "enabled_modalities": tuple(primary["modalities"]),
        "model_type": primary["type"],
        "loss_type": cfg.get("loss", {}).get("type"),
        "epochs": training.get("epochs"),
        "lr": training.get("lr"),
        "output_run_name": cfg["output"]["run_name"],
        "checkpoint_policy": cfg.get("checkpoint", cfg.get("artifacts", {})),
    }


def test_retired_raymobtime_configs_fail_fast(tmp_path: Path):
    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(ROOT / "configs/raymobtime/s008_multitask_selection.yaml")
    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(ROOT / "configs/preprocess/raymobtime_s008_cache.yaml")

    config_path = tmp_path / "retired_raymobtime.yaml"
    config_path.write_text(
        """
experiment:
  task: fusion
  objective: current_beam_selection
data:
  dataset:
    type: raymobtime_s008
model:
  primary:
    type: simple_concat_multitask_selection
    modalities: [coord]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Raymobtime s008 has been retired"):
        load_config(config_path)


def test_retired_bgam_and_viewer_configs_fail_fast(tmp_path: Path):
    retired_paths = [
        ROOT / "configs/deepsense6g_gps_lidar_bgam.yaml",
        ROOT / "configs/mmw_town_gps_lidar_bgam.yaml",
        ROOT / "configs/diagnostics/modality_visualization.yaml",
    ]
    for path in retired_paths:
        with pytest.raises(ValueError, match="retired|退役|no longer supported"):
            load_config(path)

    bgam_config = tmp_path / "retired_bgam.yaml"
    bgam_config.write_text(
        """
experiment:
  name: deepsense6g_gps_lidar_bgam_reranker
model:
  primary:
    type: gps_lidar_bgam_beam_predictor
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="BGAM"):
        load_config(bgam_config)

    viewer_config = tmp_path / "retired_viewer.yaml"
    viewer_config.write_text(
        """
diagnostics:
  visualization:
    enabled: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Viewer manifest"):
        load_config(viewer_config)
