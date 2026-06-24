import csv
import json
from pathlib import Path

import pytest

from kd_sensing.config import load_config
from kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep import (
    DEFAULT_OUTPUT_ROOT,
    REQUIRED_FAMILIES,
    FullSweepRunnerError,
    _claim_gate_reason,
    _topological_job_order,
    generate_runtime_bundle,
    generate_summary,
    load_full_sweep_manifest,
    run_job_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml"


def _read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter="\t")]


def _output_root(tmp_path: Path) -> Path:
    return tmp_path / "cnn_hybrid_jepa_visual_prior_sweep"


def test_full_sweep_manifest_schema_candidate_axes_and_seed_expansion():
    manifest = load_full_sweep_manifest(MANIFEST)
    source_families = set(manifest["families"])
    base = manifest["base_candidates"]
    expanded = manifest["expanded_candidates"]

    assert REQUIRED_FAMILIES <= source_families
    assert manifest["output_root"] == DEFAULT_OUTPUT_ROOT.as_posix()
    assert len(base) == 182
    assert len(expanded) == len(base) * 3 * 2
    assert len({candidate["variant_id"] for candidate in base}) == len(base)
    assert {row["seed"] for row in expanded} == {17, 23, 42}
    assert {row["checkpoint_selection"] for row in expanded} == {"primary", "best_top1"}

    anchors = {
        "gps_only_control",
        "image_only_resnet18_imagenet_gru",
        "image_only_patch16_mean",
        "image_only_tinyvit_5m_22k_mean",
        "patch16_mean_baseline",
        "patch16_gps_query_pool",
        "patch14_stage1_gps_query",
        "overlap_k16_s8_stage1",
        "overlap_k20_s10",
        "resnet18_layer4_tokens",
        "resnet18_layer3_layer4_tokens",
        "pooler_gps_query_k2_tokens_weighted_readout",
        "pooler_learned_query_k2_tokens",
        "pooler_self_attention_k2_tokens",
    }
    assert anchors <= {candidate["variant_id"] for candidate in base}
    required_fields = {
        "variant_id",
        "family",
        "stage_plan",
        "checkpoint_policy",
        "checkpoint_selection",
        "availability",
        "visual_encoder",
        "token_source",
        "pooler",
        "token_metadata",
        "params_metadata",
        "pooler_type",
        "pooler_output_mode",
        "k_queries",
        "token_readout_type",
        "representation_core_type",
        "strict_comparability",
        "metrics_path_template",
    }
    assert required_fields <= set(base[0])
    readout = next(candidate for candidate in base if candidate["variant_id"] == "pooler_gps_query_k2_tokens_weighted_readout")
    assert readout["pooler"]["output_mode"] == "tokens"
    assert readout["token_readout_type"] == "learned_query_weighted"
    assert readout["representation_core"]["type"] == "query_weighted_token_readout"
    learned_query = next(candidate for candidate in base if candidate["variant_id"] == "pooler_learned_query_k2_tokens")
    assert learned_query["pooler"]["type"] == "learned_query_attention"
    assert learned_query["pooler"]["output_mode"] == "tokens"
    assert learned_query["representation_core"]["type"] == "token_aware_transformer"
    self_attention = next(candidate for candidate in base if candidate["variant_id"] == "pooler_self_attention_k2_tokens")
    assert self_attention["pooler"]["type"] == "self_attention"
    assert self_attention["pooler"]["output_mode"] == "tokens"
    assert self_attention["representation_core"]["type"] == "token_aware_transformer"

    patch_sizes = {
        candidate["visual_encoder"]["patch_size"]
        for candidate in base
        if candidate["family"] == "patch_resolution_stage1"
    }
    overlap_pairs = {
        (candidate["visual_encoder"]["kernel_size"], candidate["visual_encoder"]["stride"])
        for candidate in base
        if candidate["family"] == "overlap_stage1"
    }
    assert patch_sizes == {16, 14, 12, 10, 8}
    assert overlap_pairs == {(12, 6), (14, 7), (16, 8), (20, 10), (24, 12)}

    cnn_rows = [candidate for candidate in base if candidate["family"] == "cnn_supervised_tokens"]
    assert {row["visual_encoder"].get("backbone") for row in cnn_rows} == {"resnet18", "resnet34"}
    assert {row["pretrained_source"] for row in cnn_rows} == {"scratch", "imagenet"}
    assert {row["freeze_policy"] for row in cnn_rows} == {
        "full_ft",
        "freeze_backbone_projection",
        "unfreeze_layer4",
    }

    teacher_rows = [candidate for candidate in base if candidate["family"] == "teacher_guided_stabilization"]
    assert len(teacher_rows) == 3 * 5 * 2 * 3
    assert all("teacher_guidance" in row for row in teacher_rows)

    tinyvit_rows = [candidate for candidate in base if candidate["family"] == "tinyvit_jepa_encoders"]
    assert len(tinyvit_rows) == 4
    assert {row["visual_encoder"]["encoder_type"] for row in tinyvit_rows} == {
        "tinyvit_5m_scratch_rgb",
        "tinyvit_5m_22k_rgb",
        "tinyvit_11m_scratch_rgb",
        "tinyvit_11m_22k_rgb",
    }

    image_only_rows = [candidate for candidate in base if candidate["family"] == "image_only_controls"]
    assert {row["variant_id"] for row in image_only_rows} == {
        "image_only_resnet18_imagenet_gru",
        "image_only_patch16_mean",
        "image_only_tinyvit_5m_22k_mean",
    }


def test_screening_mode_limits_candidates_best_top1_eval_and_excludes_teacher(tmp_path: Path):
    bundle = generate_runtime_bundle(output_root=_output_root(tmp_path), mode="screening", force=True)
    output_root = Path(bundle["output_root"])
    expanded = json.loads(Path(bundle["manifest_expanded_json"]).read_text(encoding="utf-8"))
    all_jobs = _read_tsv(bundle["job_paths"]["all"])
    stage1_jobs = _read_tsv(bundle["job_paths"]["stage1"])
    downstream_jobs = _read_tsv(bundle["job_paths"]["downstream"])
    teacher_jobs = _read_tsv(bundle["job_paths"]["teacher_guided"])
    reeval_jobs = _read_tsv(bundle["job_paths"]["reeval"])

    assert bundle["base_candidate_count"] == 37
    assert len(expanded) == 185
    assert {row["seed"] for row in expanded} == {42}
    assert {row["checkpoint_selection"] for row in expanded} == {"primary", "best_top1", "best_dba"}
    assert len(stage1_jobs) == 14
    assert len(downstream_jobs) == 37
    assert teacher_jobs == []
    assert len(reeval_jobs) == 148
    assert {job["checkpoint_selection"] for job in reeval_jobs} == {"best_top1", "best_dba"}
    assert {job["run_id"].rsplit("__eval_", 1)[-1] for job in reeval_jobs} == {"s31_s34", "s32_s34"}
    assert len(all_jobs) == 200

    assert {
        "tinyvit_5m_scratch_jepa_stage1",
        "tinyvit_5m_22k_jepa_stage1",
        "tinyvit_11m_scratch_jepa_stage1",
        "tinyvit_11m_22k_jepa_stage1",
    } <= {row["variant_id"] for row in expanded}
    assert {
        "image_only_resnet18_imagenet_gru",
        "image_only_patch16_mean",
        "image_only_tinyvit_5m_22k_mean",
    } <= {row["variant_id"] for row in expanded}
    assert not any(job["variant_id"].startswith("tinyvit_") for job in stage1_jobs)
    tinyvit_job = next(job for job in downstream_jobs if job["variant_id"] == "tinyvit_5m_scratch_jepa_stage1")
    cfg = load_config(tinyvit_job["config_path"])
    image_encoder = cfg["model"]["primary"]["encoders"]["image"]
    visual = image_encoder["visual_encoder"]
    assert image_encoder["type"] == "jepa_context_image"
    assert image_encoder["checkpoint_path"] == ""
    assert visual["type"] == "tinyvit_frame"
    assert visual["encoder_type"] == "tinyvit_5m_scratch_rgb"
    assert visual["freeze_backbone"] is False
    assert cfg["data"]["dataset"]["train_scenes"] == [32, 33, 34]
    assert cfg["data"]["dataset"]["test_scenes"] == [31, 32, 33, 34]
    image_only_job = next(job for job in downstream_jobs if job["variant_id"] == "image_only_patch16_mean")
    image_only_cfg = load_config(image_only_job["config_path"])
    assert image_only_cfg["model"]["primary"]["modalities"] == ["image"]
    assert image_only_cfg["model"]["primary"]["encoders"]["image"]["pooler"]["type"] == "mean"
    assert image_only_cfg["model"]["primary"]["representation_core"]["type"] == "single_gru"
    resnet_image_only_job = next(
        job for job in downstream_jobs if job["variant_id"] == "image_only_resnet18_imagenet_gru"
    )
    resnet_image_only_cfg = load_config(resnet_image_only_job["config_path"])
    assert resnet_image_only_cfg["model"]["primary"]["modalities"] == ["image"]
    assert resnet_image_only_cfg["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    readout_job = next(job for job in downstream_jobs if job["variant_id"] == "pooler_gps_query_k2_tokens_weighted_readout")
    readout_cfg = load_config(readout_job["config_path"])
    assert readout_cfg["model"]["primary"]["encoders"]["image"]["pooler"]["output_mode"] == "tokens"
    assert readout_cfg["model"]["primary"]["representation_core"]["type"] == "query_weighted_token_readout"
    learned_query_job = next(job for job in downstream_jobs if job["variant_id"] == "pooler_learned_query_k2_tokens")
    learned_query_cfg = load_config(learned_query_job["config_path"])
    assert learned_query_cfg["model"]["primary"]["encoders"]["image"]["pooler"]["type"] == "learned_query_attention"
    assert learned_query_cfg["model"]["primary"]["representation_core"]["type"] == "token_aware_transformer"
    self_attention_job = next(job for job in downstream_jobs if job["variant_id"] == "pooler_self_attention_k2_tokens")
    self_attention_cfg = load_config(self_attention_job["config_path"])
    assert self_attention_cfg["model"]["primary"]["encoders"]["image"]["pooler"]["type"] == "self_attention"
    assert self_attention_cfg["model"]["primary"]["representation_core"]["type"] == "token_aware_transformer"
    best_dba_job = next(
        job
        for job in reeval_jobs
        if job["variant_id"] == "patch14_stage1_gps_query"
        and job["checkpoint_selection"] == "best_dba"
        and job["run_id"].endswith("__eval_s32_s34")
    )
    assert best_dba_job["command"].find("/checkpoints/best.pth") >= 0
    eval_cfg = load_config(best_dba_job["config_path"])
    assert eval_cfg["data"]["dataset"]["train_scenes"] == [32, 33, 34]
    assert eval_cfg["data"]["dataset"]["test_scenes"] == [32, 33, 34]
    stage1_cfg = load_config(stage1_jobs[0]["config_path"])
    assert stage1_cfg["data"]["dataset"]["train_scenes"] == [32, 33, 34]
    assert stage1_cfg["data"]["dataset"]["test_scenes"] == [32, 33, 34]
    assert stage1_cfg["data"]["dataset"]["portion"] == 1.0
    assert stage1_cfg["data"]["dataloader"]["train_batch_size"] == 16
    assert stage1_cfg["data"]["dataloader"]["test_batch_size"] == 16
    assert stage1_cfg["data"]["dataloader"]["num_workers"] == 0
    assert stage1_cfg["training"]["epochs"] == 20
    assert stage1_cfg["training"]["cpu_threads"] == {"intra_op": 4, "inter_op": 2}
    assert cfg["training"]["cpu_threads"] == {"intra_op": 4, "inter_op": 2}
    assert (output_root / "run_full_sweep.sh").read_text(encoding="utf-8").find("--mode screening") >= 0


def test_generator_writes_configs_jobs_and_current_teacher_guidance(tmp_path: Path):
    bundle = generate_runtime_bundle(output_root=_output_root(tmp_path), force=True)
    output_root = Path(bundle["output_root"])

    assert (output_root / "manifest_expanded.json").exists()
    assert (output_root / "manifest_expanded.csv").exists()
    assert (output_root / "run_full_sweep.sh").exists()
    assert (output_root / "summarize_full_sweep.sh").exists()
    assert (output_root / "RUN_NOTE.md").exists()

    all_jobs = _read_tsv(bundle["job_paths"]["all"])
    stage1_jobs = _read_tsv(bundle["job_paths"]["stage1"])
    downstream_jobs = _read_tsv(bundle["job_paths"]["downstream"])
    teacher_jobs = _read_tsv(bundle["job_paths"]["teacher_guided"])
    reeval_jobs = _read_tsv(bundle["job_paths"]["reeval"])

    assert all_jobs
    assert stage1_jobs
    assert downstream_jobs
    assert teacher_jobs
    assert reeval_jobs
    assert all(job["command"].startswith("conda run -n kd_mm_beam ") for job in all_jobs)
    assert all("CUDA_VISIBLE_DEVICES=" not in job["command"] for job in all_jobs)
    assert {
        "job_id",
        "variant_id",
        "stage",
        "depends_on",
        "command",
        "output_dir",
        "metrics_path",
        "log_path",
    } <= set(all_jobs[0])

    ordered = _topological_job_order(all_jobs)
    order_index = {job["job_id"]: index for index, job in enumerate(ordered)}
    for job in all_jobs:
        for dependency in [item for item in job["depends_on"].split(";") if item]:
            assert order_index[dependency] < order_index[job["job_id"]]

    sample_stage1 = next(job for job in stage1_jobs if job["config_path"])
    sample_downstream = next(job for job in downstream_jobs if job["config_path"] and job["variant_id"] != "gps_only_control")
    sample_teacher_guided = next(job for job in teacher_jobs if job["stage"] == "teacher_guided")
    sample_eval = next(job for job in reeval_jobs if job["config_path"])

    for job in (sample_stage1, sample_downstream, sample_teacher_guided, sample_eval):
        cfg = load_config(job["config_path"])
        assert cfg["output"]["dir"].startswith(str(output_root))

    teacher_cfg = load_config(sample_teacher_guided["config_path"])
    assert teacher_cfg["loss"]["teacher_guidance"]["enabled"] is True
    assert teacher_cfg["loss"]["teacher_guidance"]["mode"] == "opt_in_stabilization"
    assert "distillation" not in json.dumps(teacher_cfg, sort_keys=True)


def test_runner_dry_run_gpu_dependency_status_and_cleanup_guard(tmp_path: Path):
    bundle = generate_runtime_bundle(output_root=_output_root(tmp_path), force=True)
    output_root = Path(bundle["output_root"])

    result = run_job_manifest(output_root=output_root, dry_run=True, gpu_list="0,1,2,3", max_parallel=8)

    assert result["status"] == "dry_run"
    status_rows = list(csv.DictReader(Path(result["status_csv"]).open("r", encoding="utf-8")))
    assert status_rows
    assert {row["gpu"] for row in status_rows if row["gpu"]} <= {"0", "1", "2", "3"}
    assert all(row["status"] == "dry_run" for row in status_rows)
    dry_run_text = Path(result["dry_run_commands"]).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=0 conda run -n kd_mm_beam kd-sensing-train" in dry_run_text
    snapshot = json.loads((output_root / "status/concurrency_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["gpu_list"] == ["0", "1", "2", "3"]
    assert snapshot["max_parallel"] == 8

    with pytest.raises(FullSweepRunnerError):
        run_job_manifest(output_root=ROOT / "dataset", dry_run=True, clean_output_root=True)
    with pytest.raises(FullSweepRunnerError):
        run_job_manifest(output_root=Path("/root/.container_env"), dry_run=True, clean_output_root=True)


def test_summary_outputs_full_rows_strict_ranking_pareto_and_markdown(tmp_path: Path):
    bundle = generate_runtime_bundle(output_root=_output_root(tmp_path), force=True)
    output_root = Path(bundle["output_root"])
    expanded = json.loads(Path(bundle["manifest_expanded_json"]).read_text(encoding="utf-8"))
    target_rows = [
        row
        for row in expanded
        if row["variant_id"] == "patch16_gps_query_pool" and row["checkpoint_selection"] == "primary"
    ][:3]
    assert {row["seed"] for row in target_rows} == {17, 23, 42}
    for offset, row in enumerate(target_rows):
        metrics_path = Path(row["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "top1": 0.52 + offset * 0.01,
                    "top3": 0.72 + offset * 0.01,
                    "top5": 0.83 + offset * 0.01,
                    "dba": 0.61 + offset * 0.01,
                    "beam_distance": 1.4 - offset * 0.1,
                }
            ),
            encoding="utf-8",
        )
    gate_specs = {
        "pooler_mean": {"top1": 0.40, "dba": 0.70},
        "pooler_gps_query_k2_frame": {"top1": 0.42, "dba": 0.74},
        "pooler_gps_query_k2_tokens_weighted_readout": {
            "top1": 0.44,
            "dba": 0.75,
            "diagnostics": {
                "status": "available",
                "attention_entropy": 1.1,
                "effective_patch_count": 3.0,
                "query_diversity": 0.2,
                "token_readout": {"readout_weight_mean": [0.3, 0.4, 0.3]},
            },
        },
    }
    for variant_id, metrics in gate_specs.items():
        row = next(
            item
            for item in expanded
            if item["variant_id"] == variant_id and item["checkpoint_selection"] == "primary" and item["seed"] == 17
        )
        metrics_path = Path(row["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    outputs = generate_summary(output_root=output_root)

    for path in outputs.values():
        assert Path(path).exists()
    full_rows = json.loads(Path(outputs["full_results_json"]).read_text(encoding="utf-8"))
    assert len(full_rows) == len(expanded)
    assert any(row["status"] == "missing" for row in full_rows)
    eligible = [row for row in full_rows if row["claim_eligible"]]
    assert {row["seed"] for row in eligible if row["variant_id"] == "patch16_gps_query_pool"} == {17, 23, 42}
    assert {"pooler_mean", "pooler_gps_query_k2_frame", "pooler_gps_query_k2_tokens_weighted_readout"} <= {
        row["variant_id"] for row in eligible
    }

    strict_text = Path(outputs["strict_ranking_csv"]).read_text(encoding="utf-8")
    family_text = Path(outputs["family_best_csv"]).read_text(encoding="utf-8")
    pareto_text = Path(outputs["pareto_csv"]).read_text(encoding="utf-8")
    markdown = Path(outputs["eval_summary_md"]).read_text(encoding="utf-8")
    assert "patch16_gps_query_pool" in strict_text
    assert "existing_controls" in family_text
    assert "patch16_gps_query_pool" in pareto_text
    assert "resolution" in markdown
    assert "overlap" in markdown
    assert "CNN local prior" in markdown
    assert "teacher-guided" in markdown
    readout_gate = json.loads(Path(outputs["readout_gate_json"]).read_text(encoding="utf-8"))
    gate_row = next(row for row in readout_gate if row["variant_id"] == "pooler_gps_query_k2_tokens_weighted_readout")
    assert gate_row["delta_dba_vs_pooler_gps_query_k2_frame"] == pytest.approx(0.01)
    assert gate_row["delta_dba_vs_pooler_mean"] == pytest.approx(0.05)
    assert gate_row["status"] == "pass"


def test_claim_gate_rejects_unavailable_smoke_mismatched_and_missing_rows():
    strict = {
        "split": "beambench_tableiii_input_s32_s34_train_s31_s34_test",
        "scene_set": [32, 33, 34],
        "seed": 17,
        "history_window": 5,
        "gps_input_source_window": 2,
        "prediction_horizon": 1,
        "beam_label_space": "beam64",
        "metric_profile": "beambench_linear_topk",
        "distance_metric": "linear",
        "normalization_artifact": "config_resolved",
        "difficulty_digest": "clean",
        "output_root": DEFAULT_OUTPUT_ROOT.as_posix(),
    }
    base = {
        "availability": "available",
        "status": "success",
        "strict_comparable": True,
        "smoke_only": False,
        "checkpoint_selection": "primary",
        "checkpoint_policy": "exact_reuse",
        "stage_plan": "supervised_only",
        "strict_comparability": strict,
        "seed": 17,
        "split": strict["split"],
        "metric_profile": strict["metric_profile"],
        "top1": 0.5,
        "dba": 0.6,
    }

    assert _claim_gate_reason(base) == (True, "eligible")

    cases = [
        ({"availability": "requires_component"}, "unavailable"),
        ({"status": "missing"}, "status_missing"),
        ({"smoke_only": True}, "smoke_only"),
        ({"seed": 99}, "seed_mismatch"),
        ({"split": "different_split"}, "split_mismatch"),
        ({"metric_profile": "different_metric"}, "metric_profile_mismatch"),
        ({"checkpoint_selection": "different_checkpoint"}, "checkpoint_selection_mismatch"),
        ({"top1": None}, "missing_primary_metrics"),
    ]
    for patch, expected_reason in cases:
        row = {**base, **patch}
        eligible, reason = _claim_gate_reason(row)
        assert eligible is False
        assert expected_reason in reason
