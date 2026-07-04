import argparse
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from kd_sensing.config.io import load_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/scene31/next_round/experiment_manifest.csv"
EXPECTED_P0 = {
    "proto_sampler_uniform_es40_seed3",
    "proto_sampler_uniform_es40_seed4",
    "proto_sampler_uniform_es40_seed5",
    "proto_condbtapa_weaksingle_lam005_es40_seed3",
    "proto_condbtapa_weaksingle_lam005_es40_seed4",
    "proto_condbtapa_weaksingle_lam005_es40_seed5",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed1",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed2",
    "proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed3",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed2",
    "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed3",
}
EXPECTED_P1 = {
    "proto_curriculum_easy2hard_es40_seed3",
    "proto_maskadapter_d16_condbtapa_weaksingle_es40_seed3",
}
EXPECTED_BC_P0 = {
    "proto_sampler_adaptive_gap_a05_t1_es40_seed1",
    "proto_sampler_adaptive_gap_a05_t1_es40_seed2",
    "proto_sampler_adaptive_gap_a05_t1_es40_seed3",
    "proto_sampler_uniform_beamsoft_s15_mix05_es40_seed1",
    "proto_sampler_uniform_beamsoft_s15_mix05_es40_seed2",
    "proto_sampler_uniform_beamsoft_s15_mix05_es40_seed3",
    "proto_sampler_uniform_labelsmooth005_es40_seed1",
    "proto_sampler_uniform_labelsmooth005_es40_seed2",
    "proto_sampler_uniform_labelsmooth005_es40_seed3",
    "proto_sampler_adaptive_gap_a05_t1_beamsoft_s15_mix05_es40_seed1",
    "proto_sampler_adaptive_gap_a05_t1_beamsoft_s15_mix05_es40_seed2",
    "proto_sampler_adaptive_gap_a05_t1_beamsoft_s15_mix05_es40_seed3",
}
EXPECTED_BC_P1 = {
    "proto_sampler_adaptive_loss_a05_t1_es40_seed1",
    "proto_sampler_adaptive_loss_a05_t1_es40_seed2",
    "proto_sampler_adaptive_loss_a05_t1_es40_seed3",
    "proto_sampler_adaptive_gap_a03_t1_es40_seed1",
    "proto_sampler_adaptive_gap_a03_t1_es40_seed2",
    "proto_sampler_adaptive_gap_a03_t1_es40_seed3",
    "proto_sampler_uniform_beamsoft_s10_mix05_es40_seed1",
    "proto_sampler_uniform_beamsoft_s10_mix05_es40_seed2",
    "proto_sampler_uniform_beamsoft_s10_mix05_es40_seed3",
    "proto_sampler_uniform_beamsoft_s20_mix05_es40_seed1",
    "proto_sampler_uniform_beamsoft_s20_mix05_es40_seed2",
    "proto_sampler_uniform_beamsoft_s20_mix05_es40_seed3",
}
EXPECTED_BEAMSOFT_WEAK = {
    "proto_sampler_uniform_beamsoft_s10_mix025_es40_seed1",
    "proto_sampler_uniform_beamsoft_s10_mix025_es40_seed2",
    "proto_sampler_uniform_beamsoft_s10_mix025_es40_seed3",
    "proto_sampler_uniform_beamsoft_s15_mix025_es40_seed1",
    "proto_sampler_uniform_beamsoft_s15_mix025_es40_seed2",
    "proto_sampler_uniform_beamsoft_s15_mix025_es40_seed3",
}
EXPECTED_MAGIC_CORE = {
    "proto_sampler_uniform_es40_seed1",
    "proto_sampler_uniform_es40_seed2",
    "proto_sampler_uniform_mpfr_es40_seed1",
    "proto_sampler_uniform_mpfr_es40_seed2",
    "proto_sampler_uniform_mpfr_es40_seed3",
    "proto_uniform_pattern_proto_recenter_es40_seed1",
    "proto_uniform_pattern_proto_recenter_es40_seed2",
    "proto_uniform_pattern_proto_recenter_es40_seed3",
    "proto_uniform_mpdro_tau1_es40_seed1",
    "proto_uniform_mpdro_tau1_es40_seed2",
    "proto_uniform_mpdro_tau1_es40_seed3",
}
EXPECTED_MAGIC_BASELINES = {
    "proto_sampler_uniform_jtt_sample_replay_es40_seed1",
    "proto_sampler_uniform_jtt_sample_replay_es40_seed2",
    "proto_uniform_lastlayer_retrain_es40_seed1",
    "proto_uniform_lastlayer_retrain_es40_seed2",
    "proto_uniform_groupdro_vanilla_es40_seed1",
    "proto_uniform_groupdro_vanilla_es40_seed2",
}
EXPECTED_FUNNEL_MAIN = {
    "checkpoint_selection_uniform_all_available",
    "checkpoint_selection_jtt_all_available",
    "checkpoint_selection_mpdro_all_available",
    "proto_sampler_uniform_jtt_sample_replay_es40_seed3",
    "proto_sampler_uniform_jtt_sample_replay_es40_seed4",
    "proto_sampler_uniform_jtt_sample_replay_es40_seed5",
    "proto_sampler_uniform_mvfr_score_es40_seed1",
    "proto_sampler_uniform_mvfr_score_es40_seed2",
    "proto_sampler_uniform_mvfr_score_es40_seed3",
    "proto_uniform_mpdro_tau2_lam025_es40_seed1",
    "proto_uniform_mpdro_tau2_lam025_es40_seed2",
    "proto_uniform_mpdro_tau2_lam025_es40_seed3",
    "proto_uniform_mpdro_tau4_lam025_es40_seed1",
    "proto_uniform_mpdro_tau4_lam025_es40_seed2",
    "proto_uniform_mpdro_tau4_lam025_es40_seed3",
}
EXPECTED_FUNNEL_QUICK = {
    "proto_uniform_pattern_logit_bias_seed1",
    "proto_sampler_uniform_modbias_entropy_lam001_es40_seed1",
    "proto_sampler_uniform_pattern_film_d8_es40_seed1",
    "proto_sampler_uniform_pattern_film_d16_es40_seed1",
    "proto_uniform_tta_entropy_bn_seed1",
    "proto_uniform_pbpr_fixed_seed1",
}
EXPECTED_FUNNEL_P1 = {
    "proto_uniform_mpdro_tau2_lam05_es40_seed1",
    "proto_uniform_mpdro_tau2_lam05_es40_seed2",
    "proto_uniform_mpdro_tau2_lam05_es40_seed3",
}
EXPECTED_PATTERNS = [
    "full",
    "missing_gps",
    "missing_image",
    "missing_radar",
    "missing_lidar",
    "non_gps_only",
    "gps_only",
    "image_only",
    "radar_only",
    "lidar_only",
]
EXPECTED_RETAINED_SCENE31_YAML = {
    "configs/scene31/diagnostic_gps_only_strong.yaml",
    "configs/scene31/diagnostic_image_only_strong.yaml",
    "configs/scene31/diagnostic_lidar_only_strong.yaml",
    "configs/scene31/diagnostic_radar_only_strong.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_adba.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_fusiononly.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_modw1.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed3.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed3.yaml",
    "configs/scene31/main_v3_strong_reliability_btapa_tau4.yaml",
    "configs/scene31/main_v3_strong_reliability_proto.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_fullaux.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_fullaux_l05.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_hardgps.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_seed2.yaml",
    "configs/scene31/main_v3_strong_reliability_proto_seed3.yaml",
    "configs/scene31/templates/main_v3_proto_es20_base.yaml",
    "configs/scene31/v4_weakkd_l01_t2.yaml",
    "configs/scene31/v4_weakkd_l02_t2.yaml",
    "configs/scene31/v4_weakkd_l03_t15.yaml",
}


def test_scene31_next_round_manifest_and_configs_are_consistent(tmp_path: Path):
    generator = _load_script("generate_scene31_next_round", ROOT / "scripts/generate_scene31_next_round.py")
    out_dir = tmp_path / "next_round"

    assert generator.main(["--out_dir", str(out_dir), "--overwrite", "true"]) == 0

    rows = _read_csv(out_dir / "experiment_manifest.csv")
    by_name = {row["run_name"]: row for row in rows}

    assert EXPECTED_P0 <= set(by_name)
    assert EXPECTED_P1 <= set(by_name)
    assert EXPECTED_BC_P0 <= set(by_name)
    assert EXPECTED_BC_P1 <= set(by_name)
    assert EXPECTED_BEAMSOFT_WEAK <= set(by_name)

    for row in rows:
        run_name = row["run_name"]
        config_path = _manifest_path(row["config_path"])
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg = load_config(config_path)
        training = cfg["training"]
        loss_cfg = cfg["loss"]["u_mask_beam_jepa"]

        assert config_path.exists()
        assert int(row["expected_epochs"]) == 40
        assert training["epochs"] == 40
        assert training["max_epochs"] == 40
        assert cfg["experiment"]["seed"] == int(row["seed"])
        assert cfg["experiment"]["seed"] == int(run_name.rsplit("_seed", 1)[1])
        assert cfg["output"]["run_name"] == run_name
        assert cfg["output"]["dir"] == "outputs/scene31_next_round"
        assert cfg["evaluation"]["beam_distance_circular"] is True
        assert cfg["evaluation"]["missing_patterns"]["patterns"] == EXPECTED_PATTERNS
        assert raw["model"]["primary"]["ablation_id"] == run_name

        if "sampler_uniform_curriculum" in run_name:
            assert training["missing_pattern_sampler"] == "curriculum_easy_to_hard"
            assert "epochs_11_40" in training["curriculum_schedule"]
        elif "sampler_uniform" in run_name:
            assert training["missing_pattern_sampler"] == "uniform"
            assert loss_cfg["missing_pattern_sampler"] == "uniform"

        if "sampler_adaptive" in run_name:
            assert training["missing_pattern_sampler"] == "adaptive_pattern"
            assert loss_cfg["missing_pattern_sampler"] == "adaptive_pattern"
            assert training["adaptive_temperature"] == pytest.approx(1.0)
            assert loss_cfg["adaptive_temperature"] == pytest.approx(1.0)
            assert training["adaptive_warmup_epochs"] == 3
            assert training["use_pattern_conditional_btapa"] is False
            assert loss_cfg["use_pattern_conditional_btapa"] is False
            assert training["use_weak_pattern_kd"] is False
            if "adaptive_gap" in run_name:
                assert training["adaptive_score_mode"] == "gap_to_full"
            if "adaptive_loss" in run_name:
                assert training["adaptive_score_mode"] == "loss"
            if "_a05_" in run_name:
                assert training["adaptive_alpha"] == pytest.approx(0.5)
            if "_a03_" in run_name:
                assert training["adaptive_alpha"] == pytest.approx(0.3)

        if "beamsoft" in run_name:
            assert cfg["loss"]["type"] == "beam_neighborhood_ce"
            if "mix025" in run_name:
                assert cfg["loss"]["mix_ce"] == pytest.approx(0.25)
                assert training["missing_pattern_sampler"] == "uniform"
                assert not training.get("use_pattern_conditional_btapa", False)
                assert not training.get("use_weak_pattern_kd", False)
                assert not cfg["model"]["primary"].get("use_mask_adapter", False)
            else:
                assert cfg["loss"]["mix_ce"] == pytest.approx(0.5)
            assert cfg["loss"]["circular"] is True
            if "_s10_" in run_name:
                assert cfg["loss"]["sigma"] == pytest.approx(1.0)
            if "_s15_" in run_name:
                assert cfg["loss"]["sigma"] == pytest.approx(1.5)
            if "_s20_" in run_name:
                assert cfg["loss"]["sigma"] == pytest.approx(2.0)

        if "labelsmooth005" in run_name:
            assert cfg["loss"]["type"] == "label_smoothing_ce"
            assert cfg["loss"]["smoothing"] == pytest.approx(0.05)

        expected_lambda = _lambda_from_name(run_name)
        if expected_lambda is not None:
            assert training["btapa_lambda"] == pytest.approx(expected_lambda)
            assert loss_cfg["btapa_lambda"] == pytest.approx(expected_lambda)

        if "condbtapa_weaksingle" in run_name:
            assert training["use_pattern_conditional_btapa"] is True
            assert loss_cfg["use_pattern_conditional_btapa"] is True
            assert training["btapa_apply_patterns"] == ["radar_only", "lidar_only"]
            assert loss_cfg["btapa_apply_patterns"] == ["radar_only", "lidar_only"]
            assert "gps_only" not in training["btapa_apply_patterns"]


def test_scene31_night_grid_generator_sanity(tmp_path: Path):
    generator = _load_script("generate_experiment_grid", ROOT / "scripts/generate_experiment_grid.py")
    out_dir = tmp_path / "night_grid"

    assert generator.main(["--out_dir", str(out_dir), "--seeds", "1", "2", "--overwrite", "true"]) == 0

    rows = _read_csv(out_dir / "experiment_manifest.csv")
    generated_rows = [row for row in rows if row["group"] != "baseline"]
    by_name = {row["run_name"]: row for row in generated_rows}

    assert len(rows) == 64
    assert "proto_sampler_uniform_es20_seed1" in by_name
    assert "proto_condbtapa_weaksingle_lam005_es20_seed2" in by_name
    assert "proto_maskadapter_d16_condbtapa_weaksingle_es20_seed1" in by_name

    for run_name in (
        "proto_sampler_uniform_es20_seed1",
        "proto_condbtapa_weaksingle_lam005_es20_seed2",
        "proto_maskadapter_d16_condbtapa_weaksingle_es20_seed1",
    ):
        row = by_name[run_name]
        config_path = _manifest_path(row["config_path"])
        raw_text = config_path.read_text(encoding="utf-8")
        cfg = load_config(config_path)
        loss_cfg = cfg["loss"]["u_mask_beam_jepa"]

        assert not any(
            token in raw_text
            for token in ("hist_beam", "bgam", "jepa_msac", "amr_net_gps_image", "logits_kd", "rkd", "raymobtime")
        )
        assert int(row["expected_epochs"]) == 20
        assert cfg["experiment"]["seed"] == int(run_name.rsplit("_seed", 1)[1])
        assert cfg["output"]["run_name"] == run_name
        assert cfg["model"]["primary"]["ablation_id"] == run_name

        if "sampler_uniform" in run_name:
            assert cfg["training"]["missing_pattern_sampler"] == "uniform"
            assert loss_cfg["missing_pattern_sampler"] == "uniform"
        if "lam005" in run_name:
            assert cfg["training"]["btapa_lambda"] == pytest.approx(0.05)
            assert loss_cfg["btapa_lambda"] == pytest.approx(0.05)
        if "maskadapter_d16" in run_name:
            assert cfg["model"]["primary"]["use_mask_adapter"] is True
            assert cfg["model"]["primary"]["mask_adapter_dim"] == 16


def test_scene31_magic_overnight_generator_sanity(tmp_path: Path):
    generator = _load_script("generate_scene31_magic_overnight", ROOT / "scripts/generate_scene31_magic_overnight.py")
    out_dir = tmp_path / "magic_overnight"
    output_dir = tmp_path / "magic_outputs"

    assert generator.main(["--out_dir", str(out_dir), "--output_dir", str(output_dir), "--overwrite", "true"]) == 0

    rows = _read_csv(out_dir / "experiment_manifest.csv")
    by_name = {row["run_name"]: row for row in rows}

    assert len(rows) == 17
    assert EXPECTED_MAGIC_CORE <= set(by_name)
    assert EXPECTED_MAGIC_BASELINES <= set(by_name)

    for row in rows:
        run_name = row["run_name"]
        config_path = _manifest_path(row["config_path"])
        cfg = load_config(config_path)
        training = cfg["training"]
        loss_cfg = cfg["loss"]["u_mask_beam_jepa"]

        assert config_path.exists()
        assert int(row["expected_epochs"]) == 40
        assert training["epochs"] == 40
        assert training["max_epochs"] == 40
        assert cfg["experiment"]["seed"] == int(run_name.rsplit("_seed", 1)[1])
        assert cfg["output"]["run_name"] == run_name
        assert cfg["output"]["dir"] == str(output_dir)
        assert cfg["model"]["primary"]["ablation_id"] == run_name

        if run_name.startswith("proto_sampler_uniform_es40"):
            assert training["missing_pattern_sampler"] == "uniform"
        if "mpfr" in run_name:
            assert training["missing_pattern_sampler"] == "pattern_balanced"
            assert training["failure_replay"]["mode"] == "missing_pattern_proxy"
            assert "overnight_proxy" in row["method_tags"]
        if "pattern_proto_recenter" in run_name:
            assert training["prototype_recenter"]["mode"] == "shared_balanced_training_proxy"
            assert training["lambda_proto"] == pytest.approx(0.35)
            assert training["apply_pattern_weight_to_proto"] is True
        if "mpdro_tau1" in run_name:
            assert training["mpdro"]["enabled"] is True
            assert training["mpdro"]["tau"] == pytest.approx(1.0)
            assert loss_cfg["mpdro"]["enabled"] is True
            assert training["missing_pattern_sampler"] == "pattern_balanced"


def test_scene31_funnel_generator_sanity(tmp_path: Path):
    generator = _load_script("generate_scene31_funnel", ROOT / "scripts/generate_scene31_funnel.py")
    out_dir = tmp_path / "funnel"
    output_dir = tmp_path / "funnel_outputs"

    assert generator.main(["--out_dir", str(out_dir), "--output_dir", str(output_dir), "--overwrite", "true"]) == 0

    rows = _read_csv(out_dir / "experiment_manifest.csv")
    by_name = {row["run_name"]: row for row in rows}
    assert EXPECTED_FUNNEL_MAIN <= set(by_name)
    assert EXPECTED_FUNNEL_QUICK <= set(by_name)
    assert EXPECTED_FUNNEL_P1 <= set(by_name)
    assert by_name["checkpoint_selection_uniform_all_available"]["execution_mode"] == "selection"
    assert by_name["proto_uniform_pattern_logit_bias_seed1"]["execution_mode"] == "posthoc"

    mpdro_cfg = load_config(_manifest_path(by_name["proto_uniform_mpdro_tau2_lam025_es40_seed1"]["config_path"]))
    assert mpdro_cfg["output"]["dir"] == str(output_dir)
    assert mpdro_cfg["training"]["mpdro"]["tau"] == pytest.approx(2.0)
    assert mpdro_cfg["training"]["mpdro"]["lambda_dro"] == pytest.approx(0.25)
    assert mpdro_cfg["training"]["mpdro"]["full_protection"] is True

    mvfr_cfg = load_config(_manifest_path(by_name["proto_sampler_uniform_mvfr_score_es40_seed1"]["config_path"]))
    assert mvfr_cfg["training"]["mvfr"]["enabled"] is True
    assert mvfr_cfg["training"]["mvfr"]["score_patterns"] == "missing_only"

    film_cfg = load_config(_manifest_path(by_name["proto_sampler_uniform_pattern_film_d8_es40_seed1"]["config_path"]))
    assert film_cfg["model"]["primary"]["pattern_film"]["init_identity"] is True


def test_scene31_source_yaml_surface_is_manifest_backed_or_retained():
    retained = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "configs/scene31").rglob("*.yaml")
    }
    generated_yaml = [
        path.relative_to(ROOT).as_posix()
        for root in ("night_grid", "next_round", "funnel", "magic_overnight")
        for path in (ROOT / "configs/scene31" / root).glob("*.yaml")
    ]

    assert retained == EXPECTED_RETAINED_SCENE31_YAML
    assert not generated_yaml
    for manifest_root in ("night_grid", "next_round", "funnel", "magic_overnight"):
        assert (ROOT / "configs/scene31" / manifest_root / "experiment_manifest.csv").exists()
        assert (ROOT / "configs/scene31" / manifest_root / "experiment_manifest.json").exists()


def test_scene31_runner_common_checks_manifest_train_and_eval(tmp_path):
    runner = _load_script("scene31_runner_common", ROOT / "scripts/scene31_runner_common.py")
    manifest = tmp_path / "manifest.csv"
    run_name = "proto_sampler_uniform_es40_seed1"
    _write_csv(
        manifest,
        ["run_name", "config_path", "group"],
        [{"run_name": run_name, "config_path": "configs/scene31/generated.yaml", "group": "p0"}],
    )

    assert runner.main(["manifest-value", str(manifest), run_name, "config_path"]) == 0
    assert runner.main(["manifest-value", str(manifest), "missing", "config_path"]) == 1

    root = tmp_path / "runs"
    checkpoint_dir = root / run_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pth").write_text("stub", encoding="utf-8")
    assert runner.main(["train-complete", str(root), run_name]) == 0
    assert runner.main(["train-complete", str(root), run_name, "--strict-status-checkpoint"]) == 1
    (root / run_name / "run_status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    assert runner.main(["train-complete", str(root), run_name, "--strict-status-checkpoint"]) == 0
    assert runner.main(["train-complete", str(root), "missing"]) == 1

    eval_dir = tmp_path / "fresh_eval" / run_name
    eval_dir.mkdir(parents=True)
    (eval_dir / "checkpoint_manifest.json").write_text(json.dumps({"max_batches": ""}), encoding="utf-8")
    _write_complete_eval_metrics(eval_dir / "apples_to_apples_metrics.csv")
    assert runner.main(["eval-complete", str(eval_dir), "--require-manifest"]) == 0

    (eval_dir / "checkpoint_manifest.json").write_text(json.dumps({"max_batches": 1}), encoding="utf-8")
    assert runner.main(["eval-complete", str(eval_dir), "--require-manifest"]) == 1


def test_scene31_next_round_summary_outputs_delta_and_filtered_tables(tmp_path):
    summary = _load_script("summarize_scene31_next_round", ROOT / "scripts/summarize_scene31_next_round.py")
    metrics = tmp_path / "night_grid_metrics.csv"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "summary"
    _write_csv(
        manifest,
        ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"],
        [
            {
                "run_name": "candidate_seed1",
                "group": "p0",
                "config_path": "candidate.yaml",
                "seed": "1",
                "method_tags": "candidate",
                "expected_epochs": "40",
                "priority": "high",
            },
            {
                "run_name": "candidate_seed2",
                "group": "p0",
                "config_path": "candidate.yaml",
                "seed": "2",
                "method_tags": "candidate",
                "expected_epochs": "40",
                "priority": "high",
            },
        ],
    )
    metric_rows = []
    for run_name, bump in (("candidate_seed1", 0.0), ("candidate_seed2", 0.01)):
        for pattern, value in {
            "full": 0.41 + bump,
            "avg_missing": 0.30 + bump,
            "missing_gps": 0.31 + bump,
            "missing_radar": 0.34 + bump,
            "radar_only": 0.21 + bump,
            "lidar_only": 0.12 + bump,
        }.items():
            metric_rows.append(
                {
                    "run_name": run_name,
                    "group": "p0",
                    "seed": run_name[-1],
                    "pattern": pattern,
                    "top1": str(value),
                    "status": "ok",
                }
            )
    _write_csv(metrics, ["run_name", "group", "seed", "pattern", "top1", "status"], metric_rows)

    assert summary.main(["--root", str(tmp_path / "empty_root"), "--metrics", str(metrics), "--manifest", str(manifest), "--out", str(out_dir)]) == 0

    per_run = _read_csv(out_dir / "scene31_next_round_per_run.csv")
    methods = _read_csv(out_dir / "scene31_next_round_method_mean_std.csv")
    filtered = _read_csv(out_dir / "scene31_next_round_filtered.csv")
    markdown = (out_dir / "scene31_next_round_summary.md").read_text(encoding="utf-8")

    assert "Δbalanced" in per_run[0]
    assert methods[0]["method"] == "candidate"
    assert methods[0]["balanced_mean"]
    assert filtered[0]["method"] == "candidate"
    assert "BTAPA" not in markdown
    assert "btapa_tau1" in markdown


def test_scene31_p0_fresh_summary_uses_avg_missing_primary_sort(tmp_path):
    summary = _load_script("summarize_scene31_p0_fresh_eval", ROOT / "scripts/summarize_scene31_p0_fresh_eval.py")
    metrics = tmp_path / "p0_metrics.csv"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "p0_summary"
    uniform_ckpt = tmp_path / "uniform_best.pth"
    lam_ckpt = tmp_path / "lam_best.pth"
    uniform_ckpt.write_text("stub", encoding="utf-8")
    lam_ckpt.write_text("stub", encoding="utf-8")
    _write_csv(
        manifest,
        ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"],
        [
            {
                "run_name": "proto_sampler_uniform_es40_seed3",
                "group": "p0",
                "config_path": "uniform.yaml",
                "seed": "3",
                "method_tags": "sampler,uniform,es40",
                "expected_epochs": "40",
                "priority": "high",
            },
            {
                "run_name": "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1",
                "group": "p0",
                "config_path": "lam.yaml",
                "seed": "1",
                "method_tags": "sampler,uniform,condbtapa,weak_single,lambda_0.025,es40",
                "expected_epochs": "40",
                "priority": "high",
            },
        ],
    )
    metric_rows = []
    runs = {
        "proto_sampler_uniform_es40_seed3": (uniform_ckpt, {"full": 0.4200, "missing_gps": 0.3000, "missing_radar": 0.3200, "radar_only": 0.1000, "lidar_only": 0.0900}),
        "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1": (
            lam_ckpt,
            {"full": 0.4100, "missing_gps": 0.3300, "missing_radar": 0.3400, "radar_only": 0.2300, "lidar_only": 0.1300},
        ),
    }
    for run_name, (checkpoint, values) in runs.items():
        avg_missing = sum(value for key, value in values.items() if key != "full") / 4
        for pattern, value in {**values, "avg_missing": avg_missing}.items():
            metric_rows.append(
                {
                    "run_name": run_name,
                    "pattern": pattern,
                    "top1": str(value),
                    "status": "ok",
                    "checkpoint_path": str(checkpoint),
                    "max_batches": "",
                }
            )
    _write_csv(metrics, ["run_name", "pattern", "top1", "status", "checkpoint_path", "max_batches"], metric_rows)

    assert summary.main(["--root", str(tmp_path / "empty_root"), "--metrics", str(metrics), "--manifest", str(manifest), "--out", str(out_dir)]) == 0

    per_run = _read_csv(out_dir / "p0_per_run.csv")
    methods = _read_csv(out_dir / "p0_method_mean_std.csv")
    delta = _read_csv(out_dir / "p0_delta_vs_proto.csv")
    filtered = _read_csv(out_dir / "p0_filtered.csv")
    rank = (out_dir / "p0_rank_by_avg_missing.md").read_text(encoding="utf-8")

    assert {row["method"] for row in per_run} == {
        "proto_sampler_uniform_es40",
        "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40",
    }
    assert "overall_mean" in per_run[0]
    assert methods[0]["method"] == "proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40"
    assert "overall_mean_mean" in methods[0]
    assert "delta_overall_mean" in delta[0]
    assert "radar_only" not in filtered[0]["unmet_conditions"]
    assert "lidar_only" not in filtered[0]["unmet_conditions"]
    assert "| proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40 |" in rank


def test_scene31_bc_summary_outputs_uniform_delta_and_optional_metrics(tmp_path):
    summary = _load_script("summarize_scene31_bc_next", ROOT / "scripts/summarize_scene31_bc_next.py")
    metrics = tmp_path / "bc_metrics.csv"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "bc_summary"
    _write_csv(
        manifest,
        ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"],
        [
            {
                "run_name": "proto_sampler_adaptive_gap_a05_t1_es40_seed1",
                "group": "b_p0",
                "config_path": "adaptive1.yaml",
                "seed": "1",
                "method_tags": "sampler,adaptive",
                "expected_epochs": "40",
                "priority": "high",
            },
            {
                "run_name": "proto_sampler_adaptive_gap_a05_t1_es40_seed2",
                "group": "b_p0",
                "config_path": "adaptive2.yaml",
                "seed": "2",
                "method_tags": "sampler,adaptive",
                "expected_epochs": "40",
                "priority": "high",
            },
        ],
    )
    metric_rows = []
    for run_name, bump in (("proto_sampler_adaptive_gap_a05_t1_es40_seed1", 0.0), ("proto_sampler_adaptive_gap_a05_t1_es40_seed2", 0.01)):
        for pattern, value in {
            "full": 0.4200 + bump,
            "missing_gps": 0.3100 + bump,
            "missing_radar": 0.3300 + bump,
            "missing_gps_radar": 0.2500 + bump,
            "radar_only": 0.2200 + bump,
            "lidar_only": 0.1200 + bump,
        }.items():
            metric_rows.append(
                {
                    "run_name": run_name,
                    "pattern": pattern,
                    "top1": str(value),
                    "top3": str(value + 0.2),
                    "top5": str(value + 0.3),
                    "within_3": str(value + 0.1),
                    "mae": str(4.0 - bump),
                    "status": "ok",
                }
            )
    _write_csv(metrics, ["run_name", "pattern", "top1", "top3", "top5", "within_3", "mae", "status"], metric_rows)

    assert summary.main(["--root", str(tmp_path / "empty_root"), "--metrics", str(metrics), "--manifest", str(manifest), "--out", str(out_dir)]) == 0

    per_run = _read_csv(out_dir / "bc_per_run.csv")
    methods = _read_csv(out_dir / "bc_method_mean_std.csv")
    delta = _read_csv(out_dir / "bc_delta_vs_uniform.csv")
    markdown = (out_dir / "bc_rank_by_avg_missing.md").read_text(encoding="utf-8")
    proximity = (out_dir / "bc_rank_by_beam_proximity.md").read_text(encoding="utf-8")
    conclusion = (out_dir / "bc_conservative_conclusion.md").read_text(encoding="utf-8")

    assert per_run[0]["method"] == "proto_sampler_adaptive_gap_a05_t1_es40"
    assert "delta_vs_uniform_avg_missing" in per_run[0]
    assert "avg_missing_top1" in per_run[0]
    assert "miss1_top1" in per_run[0]
    assert "miss2_top1" in per_run[0]
    assert "miss3_top1" in per_run[0]
    assert "miss1_within_3" in per_run[0]
    assert "miss2_mae" in per_run[0]
    assert "avg_missing_top3" in per_run[0]
    assert "overall_mean_top3" in per_run[0]
    assert "avg_missing_within_3" in per_run[0]
    assert "avg_missing_mae" in per_run[0]
    assert methods[0]["method"] == "proto_sampler_adaptive_gap_a05_t1_es40"
    assert "avg_missing_top1_mean" in methods[0]
    assert "overall_mean_within_3_mean" in methods[0]
    assert "delta_vs_uniform_overall_mean_mean" in methods[0]
    assert delta[0]["method"] == "proto_sampler_adaptive_gap_a05_t1_es40"
    assert "delta_vs_uniform_avg_missing_within_3" in delta[0]
    assert "delta_vs_uniform_avg_missing" in markdown
    assert "avg_missing_within_3" in proximity
    assert "Current exact-Top1 winner:" in conclusion
    mapping = json.loads((out_dir / "missing_bucket_mapping.json").read_text(encoding="utf-8"))
    assert mapping["full"]["missing_count"] == 0
    assert mapping["missing_gps"]["missing_count"] == 1
    assert mapping["missing_gps_radar"]["missing_count"] == 2
    assert mapping["radar_only"]["missing_count"] == 3
    assert (out_dir / "bc_rank_by_miss1_top1.md").exists()
    assert (out_dir / "bc_rank_by_miss2_top1.md").exists()
    assert (out_dir / "bc_rank_by_miss3_top1.md").exists()


def test_scene31_bc_summary_warns_for_empty_missing_bucket(tmp_path):
    summary = _load_script("summarize_scene31_bc_next", ROOT / "scripts/summarize_scene31_bc_next.py")
    metrics = tmp_path / "metrics.csv"
    out_dir = tmp_path / "summary"
    _write_csv(
        metrics,
        ["run_name", "pattern", "top1", "status"],
        [
            {"run_name": "tiny_seed1", "pattern": "full", "top1": "0.4", "status": "ok"},
            {"run_name": "tiny_seed1", "pattern": "missing_gps", "top1": "0.3", "status": "ok"},
        ],
    )

    assert summary.main(["--metrics", str(metrics), "--root", str(tmp_path / "empty"), "--out", str(out_dir)]) == 0

    sanity = (out_dir / "bc_sanity_check.md").read_text(encoding="utf-8")
    assert "miss2" in sanity
    assert "miss3" in sanity


def test_scene31_missing_aware_checkpoint_selection_outputs_links_and_scores(tmp_path):
    selector = _load_script("select_missing_aware_checkpoint", ROOT / "scripts/select_missing_aware_checkpoint.py")
    root = tmp_path / "runs"
    run = "proto_sampler_uniform_es40_seed1"
    ckpt_dir = root / run / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch in (1, 2):
        ckpt = ckpt_dir / f"epoch_{epoch}.pth"
        ckpt.write_text(f"checkpoint {epoch}", encoding="utf-8")
        (ckpt_dir / f"epoch_{epoch}.pth.json").write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
    metrics = tmp_path / "selection_metrics.csv"
    rows = []
    values = {
        1: {"full": 0.50, "missing_gps": 0.20, "missing_gps_radar": 0.18, "radar_only": 0.10},
        2: {"full": 0.45, "missing_gps": 0.30, "missing_gps_radar": 0.28, "radar_only": 0.22},
    }
    for epoch, by_pattern in values.items():
        for pattern, top1 in by_pattern.items():
            rows.append(
                {
                    "run_name": run,
                    "checkpoint_epoch": str(epoch),
                    "pattern": pattern,
                    "top1": str(top1),
                    "status": "ok",
                }
            )
    _write_csv(metrics, ["run_name", "checkpoint_epoch", "pattern", "top1", "status"], rows)
    out_dir = tmp_path / "selection"

    assert selector.main(["--root", str(root), "--runs", run, "--metrics", str(metrics), "--out", str(out_dir)]) == 0

    summary_rows = _read_csv(out_dir / "checkpoint_selection_summary.csv")
    by_rule = {row["rule"]: row for row in summary_rows}
    assert by_rule["best_full_val"]["selected_epoch"] == "1"
    assert by_rule["best_avg_missing_val"]["selected_epoch"] == "2"
    assert by_rule["best_bucket_balanced_val"]["selected_epoch"] == "2"
    assert (out_dir / run / "selected_checkpoints" / "best_avg_missing_val" / "best.ckpt").exists()


def test_scene31_funnel_summary_writes_required_outputs_and_promotion_labels(tmp_path):
    summary = _load_script("summarize_scene31_funnel", ROOT / "scripts/summarize_scene31_funnel.py")
    root = tmp_path / "funnel_root"
    out_dir = tmp_path / "summary"
    metrics = tmp_path / "metrics.csv"
    manifest = tmp_path / "manifest.csv"
    selection_dir = root / "checkpoint_selection" / "checkpoint_selection_uniform_all_available"
    selection_dir.mkdir(parents=True)
    _write_csv(
        selection_dir / "checkpoint_selection_summary.csv",
        ["run", "rule", "selected_epoch", "score"],
        [{"run": "proto_sampler_uniform_es40_seed1", "rule": "best_avg_missing_val", "selected_epoch": "2", "score": "0.3"}],
    )
    _write_csv(
        manifest,
        ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority", "execution_mode"],
        [
            {"run_name": "proto_sampler_uniform_es40_seed1", "group": "main", "config_path": "", "seed": "1", "method_tags": "uniform", "expected_epochs": "40", "priority": "high", "execution_mode": "train"},
            {"run_name": "proto_uniform_pattern_logit_bias_seed1", "group": "quick", "config_path": "", "seed": "1", "method_tags": "quick", "expected_epochs": "40", "priority": "medium", "execution_mode": "posthoc"},
        ],
    )
    rows = []
    values = {
        "proto_sampler_uniform_es40_seed1": {"full": 0.4216, "missing_gps": 0.28, "missing_gps_radar": 0.25, "radar_only": 0.18},
        "proto_uniform_pattern_logit_bias_seed1": {"full": 0.4220, "missing_gps": 0.32, "missing_gps_radar": 0.30, "radar_only": 0.26},
    }
    for run_name, by_pattern in values.items():
        for pattern, top1 in by_pattern.items():
            rows.append(
                {
                    "run_name": run_name,
                    "pattern": pattern,
                    "top1": str(top1),
                    "within_3": str(top1 + 0.3),
                    "mae": str(5.0 - top1),
                    "status": "ok",
                }
            )
    _write_csv(metrics, ["run_name", "pattern", "top1", "within_3", "mae", "status"], rows)

    assert summary.main(["--root", str(root), "--metrics", str(metrics), "--manifest", str(manifest), "--out", str(out_dir)]) == 0

    methods = _read_csv(out_dir / "funnel_method_mean_std.csv")
    quick = next(row for row in methods if row["method"] == "proto_uniform_pattern_logit_bias")
    assert "promote_to_full_seeds" in quick["main_read"]
    assert (out_dir / "funnel_per_run.csv").exists()
    assert (out_dir / "funnel_delta_vs_uniform.csv").exists()
    assert (out_dir / "rank_by_avg_missing_top1.md").exists()
    assert (out_dir / "rank_by_miss2_top1.md").exists()
    assert _read_csv(out_dir / "checkpoint_selection_summary.csv")[0]["selected_epoch"] == "2"
    conclusion = (out_dir / "funnel_conclusion.txt").read_text(encoding="utf-8")
    assert "Quick screens promoted to full seeds:" in conclusion
    assert "proto_uniform_pattern_logit_bias" in conclusion


def test_scene31_beamsoft_weak_summary_wrapper_writes_standard_names(tmp_path):
    summary = _load_script("summarize_scene31_beamsoft_weak", ROOT / "scripts/summarize_scene31_beamsoft_weak.py")
    metrics = tmp_path / "metrics.csv"
    out_dir = tmp_path / "summary"
    rows = []
    for run_name, base in (
        ("proto_sampler_uniform_es40_seed3", 0.30),
        ("proto_sampler_uniform_beamsoft_s10_mix025_es40_seed1", 0.31),
    ):
        for pattern, value in {
            "full": base + 0.10,
            "missing_gps": base,
            "missing_radar": base + 0.01,
            "radar_only": base - 0.02,
            "lidar_only": base - 0.03,
        }.items():
            rows.append(
                {
                    "run_name": run_name,
                    "pattern": pattern,
                    "top1": str(value),
                    "top3": str(value + 0.2),
                    "top5": str(value + 0.3),
                    "within_3": str(value + 0.1),
                    "mae": str(4.0 - value),
                    "status": "ok",
                }
            )
    _write_csv(metrics, ["run_name", "pattern", "top1", "top3", "top5", "within_3", "mae", "status"], rows)

    assert summary.main(["--metrics", str(metrics), "--bc-root", str(tmp_path / "empty_bc"), "--weak-root", str(tmp_path / "empty_weak"), "--uniform-root", str(tmp_path / "empty_uniform"), "--out", str(out_dir)]) == 0

    assert (out_dir / "per_run.csv").exists()
    assert (out_dir / "method_mean_std.csv").exists()
    assert (out_dir / "delta_vs_uniform.csv").exists()
    assert (out_dir / "rank_by_avg_missing_top1.md").exists()
    conclusion = (out_dir / "conservative_conclusion.md").read_text(encoding="utf-8")
    assert "Uniform vs proto_sampler_uniform_beamsoft_s10_mix025_es40:" in conclusion


def test_scene31_bc_launcher_help_and_syntax():
    scripts = [
        (ROOT / "scripts/run_scene31_bc_next.sh", "baselines"),
        (ROOT / "scripts/run_scene31_beamsoft_weak.sh", "s10_mix025"),
        (ROOT / "scripts/run_scene31_bc_apples_eval.sh", "uniform-root"),
        (ROOT / "scripts/run_scene31_funnel.sh", "mild_mpdro"),
        (ROOT / "scripts/run_scene31_magic_overnight.sh", "overnight_core"),
    ]
    subprocess.run(["bash", "-n", str(ROOT / "scripts/scene31_runner_common.sh")], check=True)
    for script, marker in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
        help_result = subprocess.run(["bash", str(script), "--help"], check=True, text=True, capture_output=True)
        assert marker in help_result.stdout


def test_scene31_next_round_balanced_formula_matches_existing_analyzer():
    summary = _load_script("summarize_scene31_next_round", ROOT / "scripts/summarize_scene31_next_round.py")
    analyzer = _load_script("analyze_night_grid", ROOT / "scripts/analyze_night_grid.py")
    row = {
        "full": 0.4100,
        "avg_missing": 0.3000,
        "missing_gps": 0.3100,
        "missing_radar": 0.3200,
        "radar_only": 0.2100,
        "lidar_only": 0.1200,
    }
    proto = dict(summary.PROTO_REFERENCE)
    analyzer_row = {f"{key}_top1": value for key, value in row.items()}
    analyzer_proto = {f"{key}_top1": value for key, value in proto.items()}
    args = argparse.Namespace(
        radar_weight=0.25,
        lidar_weight=0.25,
        missing_gps_penalty=0.5,
        missing_radar_penalty=0.5,
        full_penalty=0.25,
    )

    assert summary._balanced(row, proto) == pytest.approx(analyzer._balanced_score(analyzer_row, analyzer_proto, args))


def _lambda_from_name(run_name: str) -> float | None:
    if "lam0025" in run_name:
        return 0.025
    if "lam005" in run_name:
        return 0.05
    if "lam001" in run_name:
        return 0.01
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_complete_eval_metrics(path: Path) -> None:
    rows = []
    for pattern in ("full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only"):
        rows.append(
            {
                "pattern": pattern,
                "top1": "0.3",
                "top3": "0.5",
                "top5": "0.6",
                "within_3": "0.4",
                "mae": "4.0",
                "status": "ok",
            }
        )
    _write_csv(path, ["pattern", "top1", "top3", "top5", "within_3", "mae", "status"], rows)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
