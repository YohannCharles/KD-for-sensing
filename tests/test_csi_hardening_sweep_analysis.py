import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_csi_hardening_sweep.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("analyze_csi_hardening_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_csi_hardening_sweep_analysis_outputs_metrics_and_csv(tmp_path: Path):
    module = _load_script_module()
    runs_root = tmp_path / "runs"
    _write_gate_runs(runs_root)
    clean = runs_root / "csi_A0_clean_full_teacher"
    variant = runs_root / "csi_B5_mild_hardening_combo"
    _write_run(
        clean,
        run_name="csi_A0_clean_full_teacher",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
    )
    _write_run(
        variant,
        run_name="csi_B5_mild_hardening_combo",
        val_acc=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.75, *([0.8] * 8)],
        hardening=True,
        diagnostics=True,
    )
    out = tmp_path / "analysis"

    result = module.main(
        [
            "--runs_root",
            str(runs_root),
            "--pattern",
            "csi_*",
            "--clean_teacher_run",
            "csi_A0_clean_full_teacher",
            "--out",
            str(out),
        ]
    )

    assert result["runs"] == 7
    summary = _read_csv(out / "summary.csv")
    ranked = _read_csv(out / "ranked_candidates.csv")
    variant_row = next(row for row in summary if row["run_name"] == "csi_B5_mild_hardening_combo")
    assert variant_row["E50"] == "6"
    assert variant_row["E80"] == "9"
    assert variant_row["E90"] == "11"
    assert variant_row["is_destructive"] == "False"
    assert variant_row["is_slow_high_ceiling"] == "True"
    assert variant_row["pilot_noise_scale_valid"] == "True"
    assert variant_row["full_sweep_status"] == "valid"
    assert variant_row["candidate_eligible"] == "True"
    assert variant_row["hardening_design_failed"] == "False"
    assert ranked[0]["run_name"] == "csi_B5_mild_hardening_combo"
    assert (out / "analysis_metadata.json").exists()
    assert (out / "learning_curves.png").exists()
    assert (out / "ceiling_gap_vs_E90_ratio.png").exists()


def test_csi_hardening_sweep_analysis_marks_missing_legacy_diagnostics_invalid(tmp_path: Path):
    module = _load_script_module()
    runs_root = tmp_path / "outputs/csi_hardening_matrix_20260520_164406/Town10_skybridge_seed24"
    clean = runs_root / "csi_A0_clean_full_teacher"
    variant = runs_root / "csi_A1_mild_pilot_estimation"
    _write_run(clean, run_name="csi_A0_clean_full_teacher", val_acc=[0.6, 0.8], hardening=False, diagnostics=False)
    _write_run(
        variant,
        run_name="csi_A1_mild_pilot_estimation",
        val_acc=[0.1, 0.14],
        hardening=False,
        diagnostics=False,
        csi_estimation={"mode": "physical", "pilot_len": 16, "pilot_power": 1.0, "noise_var": 0.01},
    )
    out = tmp_path / "analysis"

    module.main(
        [
            "--runs_root",
            str(runs_root),
            "--pattern",
            "csi_*",
            "--clean_teacher_run",
            "csi_A0_clean_full_teacher",
            "--out",
            str(out),
        ]
    )

    summary = _read_csv(out / "summary.csv")
    ranked = _read_csv(out / "ranked_candidates.csv")
    assert {row["full_sweep_status"] for row in summary} == {"invalid_due_to_missing_debug_diagnostics"}
    assert all(row["invalid_reason"] for row in summary)
    assert ranked == []
    metadata = json.loads((out / "analysis_metadata.json").read_text(encoding="utf-8"))
    assert metadata["legacy_invalid_sweep_isolated"] is True


def test_csi_hardening_sweep_analysis_excludes_invalid_pilot_noise(tmp_path: Path):
    module = _load_script_module()
    runs_root = tmp_path / "runs"
    _write_gate_runs(runs_root)
    _write_run(
        runs_root / "csi_A0_clean_full_teacher",
        run_name="csi_A0_clean_full_teacher",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
    )
    _write_run(
        runs_root / "csi_A1_mild_pilot_estimation",
        run_name="csi_A1_mild_pilot_estimation",
        val_acc=[0.05, 0.1, 0.15, *([0.2] * 17)],
        hardening=False,
        diagnostics=True,
        csi_estimation={"mode": "est_snr", "snr_db": 30.0, "train_snr_min_db": 25.0, "train_snr_max_db": 35.0},
        pilot_ratio=10.0,
        pilot_snr=[30.0],
    )
    out = tmp_path / "analysis"

    module.main(
        [
            "--runs_root",
            str(runs_root),
            "--pattern",
            "csi_*",
            "--clean_teacher_run",
            "csi_A0_clean_full_teacher",
            "--out",
            str(out),
        ]
    )

    summary = _read_csv(out / "summary.csv")
    ranked = _read_csv(out / "ranked_candidates.csv")
    a1 = next(row for row in summary if row["run_name"] == "csi_A1_mild_pilot_estimation")
    assert a1["pilot_noise_scale_valid"] == "False"
    assert a1["invalid_reason"] == "invalid_due_to_pilot_noise_scale"
    assert all(row["run_name"] != "csi_A1_mild_pilot_estimation" for row in ranked)


def _write_gate_runs(runs_root: Path) -> None:
    _write_run(
        runs_root / "csi_debug_A0_original",
        run_name="csi_debug_A0_original",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
        matrix_role="A0_original",
    )
    _write_run(
        runs_root / "csi_debug_A0_clone_generated",
        run_name="csi_debug_A0_clone_generated",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
        matrix_role="A0_clone_generated",
        parity_passed=True,
    )
    _write_run(
        runs_root / "csi_debug_A0_clone_pilot_disabled",
        run_name="csi_debug_A0_clone_pilot_disabled",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
        matrix_role="A0_clone_pilot_disabled",
        csi_estimation={"enabled": False, "mode": "est_snr", "snr_db": 30.0},
    )
    _write_run(
        runs_root / "csi_debug_C1_view_gate_warmup_only",
        run_name="csi_debug_C1_view_gate_warmup_only",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
        matrix_role="C1_view_gate_warmup_only",
    )
    _write_run(
        runs_root / "csi_debug_C2_no_internal_gru_only",
        run_name="csi_debug_C2_no_internal_gru_only",
        val_acc=[0.1, 0.5, *([0.8] * 18)],
        hardening=False,
        diagnostics=True,
        matrix_role="C2_no_internal_gru_only",
    )


def _write_run(
    run_dir: Path,
    *,
    run_name: str,
    val_acc: list[float],
    hardening: bool,
    diagnostics: bool,
    csi_estimation: dict | None = None,
    matrix_role: str | None = None,
    parity_passed: bool | None = None,
    pilot_ratio: float = 0.0,
    pilot_snr: list[float] | None = None,
) -> None:
    run_dir.mkdir(parents=True)
    train_log = {
        "val_acc": val_acc,
        "val_adba": [0.1 + idx * 0.01 for idx in range(len(val_acc))],
        "epoch_logs": [{"epoch": idx + 1, "val_acc": value} for idx, value in enumerate(val_acc)],
    }
    if diagnostics:
        train_log["csi_first_batch_diagnostics"] = [
            {
                "source": "train",
                "pilot": {
                    "pilot_identity_max_abs": 0.0 if pilot_ratio == 0.0 else 0.1,
                    "sigma_e2": pilot_ratio,
                    "h_power_mean": 1.0,
                    "noise_power_mean": pilot_ratio,
                    "h_hat_power_mean": 1.0 + pilot_ratio,
                    "noise_power_signal_ratio": pilot_ratio,
                    **({"snr_db": pilot_snr} if pilot_snr is not None else {}),
                },
            }
        ]
    (run_dir / "train_log.json").write_text(json.dumps(train_log), encoding="utf-8")
    hardening_yaml = (
        "    csi_hardening:\n"
        "      enabled: true\n"
        "      common_phase:\n"
        "        enabled: true\n"
        if hardening
        else ""
    )
    csi_estimation = csi_estimation or {"mode": "none"}
    estimation_yaml = "\n".join(f"          {key}: {str(value).lower() if isinstance(value, bool) else value}" for key, value in csi_estimation.items())
    matrix_role = matrix_role or run_name.replace("csi_", "")
    (run_dir / "final_config.yaml").write_text(
        "experiment:\n"
        f"  name: {run_name}\n"
        "  seed: 42\n"
        "debug:\n"
        f"  matrix_role: {matrix_role}\n"
        "  pilot_scaling_config_version: fixed_estimation_snr_v1\n"
        "data:\n"
        "  dataset:\n"
        f"{hardening_yaml}"
        "model:\n"
        "  modalities: [csi]\n"
        "  primary:\n"
        "    encoders:\n"
        "      csi:\n"
        "        type: pilot_dual_view_csi\n"
        "        csi_estimation:\n"
        f"{estimation_yaml}\n"
        "output:\n"
        f"  run_name: {run_name}\n",
        encoding="utf-8",
    )
    if parity_passed is not None:
        (run_dir / "config_diff.json").write_text(
            json.dumps({"parity_passed": parity_passed, "status": "passed" if parity_passed else "failed"}),
            encoding="utf-8",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
