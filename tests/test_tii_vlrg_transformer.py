from pathlib import Path

from kd_sensing.baselines.tii_vlrg_transformer import build_manifest, build_summary_row, run_reproduction
from kd_sensing.cli.tii_vlrg_transformer import build_parser
from kd_sensing.config import load_config
from kd_sensing.engine.optim import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_manifest_records_missing_artifacts_and_conda_commands(tmp_path: Path):
    manifest = build_manifest(
        {
            "source_repo": tmp_path / "missing_repo",
            "checkpoint_path": tmp_path / "missing.pt",
            "prediction_path": tmp_path / "missing_predictions.csv",
            "output_root": tmp_path / "run",
        },
        dry_run=True,
    )

    assert manifest["status"] == "pending"
    assert manifest["dry_run"]["will_execute"] is False
    assert manifest["dry_run"]["commands"][0]["command"][:4] == ["conda", "run", "-n", "kd_mm_beam"]
    assert any("source_repo unavailable" in warning for warning in manifest["warnings"])


def test_metrics_csv_import_writes_external_reference_summary(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("overall_clean,P0,P1,sample_count\n0.72,0.80,0.64,10\n", encoding="utf-8")

    result = run_reproduction(
        None,
        overrides={
            "source_repo": repo,
            "source_commit": "abc123",
            "checkpoint_path": checkpoint,
            "metrics_path": metrics,
        },
        output_root=tmp_path / "out",
        dry_run=True,
    )

    summary = result["summary_row"]
    assert Path(result["manifest_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert summary["overall_clean"] == 0.72
    assert summary["overall_p0_p5_mean"] == 0.72
    assert summary["strict_comparability"] == "not_comparable"
    assert summary["comparison_scope"] == "external_reference"
    assert summary["strict_ranking_eligible"] is False
    assert summary["source_commit"] == "abc123"
    assert summary["checkpoint_sha256"]


def test_prediction_csv_import_computes_official_dba(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prediction = tmp_path / "predictions.csv"
    prediction.write_text("label,top1,top2,top3\n10,10,11,12\n20,22,20,21\n", encoding="utf-8")
    manifest = build_manifest(
        {
            "source_repo": repo,
            "prediction_path": prediction,
            "metric_profile": "beambench_linear_topk",
            "split": "deepsense6g_s32_s34_train_s31_s34_eval",
            "gps_source_window": "current",
            "seed": 42,
            "difficulty_digest": "clean",
        },
        output_root=tmp_path / "out",
    )

    summary = build_summary_row(manifest)

    assert summary["sample_count"] == 2
    assert summary["overall_clean"] > 0.8
    assert summary["strict_comparability"] == "strict"


def test_cli_help_declares_manifest_inputs():
    help_text = build_parser().format_help()

    assert "--config" in help_text
    assert "--metrics-path" in help_text
    assert "--execute" in help_text


def test_tii_vlrg_local_baseline_config_builds_without_external_artifacts():
    cfg = load_config(ROOT / "configs/fusion/tii_vlrg_transformer_baseline.yaml")
    primary = cfg["model"]["primary"]

    assert primary["type"] == "modular_sequence"
    assert primary["modalities"] == ["image", "radar", "gps", "lidar"]
    assert primary["encoders"]["image"]["pretrained"] is False
    assert primary["encoders"]["image"]["weights"] is None
    assert primary["representation_core"]["type"] == "token_transformer"
    assert primary["paper_metadata"]["baseline_scope"] == "local_experimental_baseline"

    model = build_model(primary)
    metadata = model.training_strategy_metadata()
    assert metadata["model_group"] == "tii_vlrg_transformer_baseline"
    assert metadata["baseline_scope"] == "local_experimental_baseline"


def test_execute_runs_synthetic_external_command_and_imports_prediction(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prediction = tmp_path / "out" / "predictions" / "tii_predictions.csv"
    code = (
        "from pathlib import Path; "
        f"path = Path({str(prediction)!r}); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('label,top1,top2,top3\\n10,10,11,12\\n', encoding='utf-8')"
    )

    result = run_reproduction(
        None,
        overrides={
            "source_repo": repo,
            "prediction_path": prediction,
            "metric_profile": "beambench_linear_topk",
            "split": "deepsense6g_s32_s34_train_s31_s34_eval",
            "gps_source_window": "current",
            "seed": 42,
            "difficulty_digest": "clean",
            "external_commands": [
                {
                    "stage": "infer",
                    "command": ["conda", "run", "-n", "kd_mm_beam", "python", "-c", code],
                }
            ],
        },
        output_root=tmp_path / "out",
        dry_run=False,
        execute=True,
    )

    assert result["manifest"]["status"] == "imported"
    assert result["manifest"]["execution"]["status"] == "complete"
    assert Path(result["manifest"]["execution"]["records"][0]["stdout_path"]).exists()
    assert result["summary_row"]["overall_clean"] == 1.0


def test_failed_execute_blocks_summary_even_if_old_prediction_exists(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    prediction = tmp_path / "old_predictions.csv"
    prediction.write_text("label,top1,top2,top3\n10,10,11,12\n", encoding="utf-8")

    result = run_reproduction(
        None,
        overrides={
            "source_repo": repo,
            "prediction_path": prediction,
            "external_commands": [
                {
                    "stage": "infer",
                    "command": ["conda", "run", "-n", "kd_mm_beam", "python", "-c", "raise SystemExit(3)"],
                }
            ],
        },
        output_root=tmp_path / "out",
        dry_run=False,
        execute=True,
    )

    assert result["manifest"]["status"] == "blocked"
    assert "summary_row" not in result
