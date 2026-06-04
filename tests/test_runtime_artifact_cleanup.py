from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kd_sensing.config.io import dump_config
from kd_sensing.diagnostics.runtime_artifact_cleanup import (
    apply_cleanup_manifest,
    build_cleanup_manifest,
    write_cleanup_manifest,
)


NOW = dt.datetime(2026, 5, 25, 12, 0, tzinfo=dt.timezone.utc)


def _base_cfg(name: str) -> dict:
    return {
        "experiment": {"name": name, "task": "fusion", "objective": "beam", "seed": 7},
        "data": {"dataset": {"type": "mmw"}},
        "model": {"student": {"modalities": ["image", "gps"]}},
        "output": {"run_name": name},
    }


def _write_started_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    cfg = _base_cfg(name)
    dump_config(cfg, run_dir / "final_config.yaml")
    dump_config(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "startup_summary.json").write_text(json.dumps({"device": "cpu"}), encoding="utf-8")
    return run_dir


def _records_by_rule(manifest: dict, section: str) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for record in manifest[section]:
        for rule_id in record.get("matched_rules", [record.get("rule_id")]):
            records.setdefault(rule_id, []).append(record)
    return records


def test_cleanup_manifest_records_outputs_other_and_checkpoint_retention(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "other" / "Town10_crossroad_seed24", "complete_run")
    (run_dir / "metrics.json").write_text(json.dumps({"val_adba": 0.5}), encoding="utf-8")
    (run_dir / "train_log.json").write_text(json.dumps({"epoch_logs": [{"epoch": 1}]}), encoding="utf-8")
    (run_dir / "training_outputs.npz").write_bytes(b"outputs")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pth").write_bytes(b"best weights")
    (checkpoint_dir / "best.pth.json").write_text(
        json.dumps({"selection_metric": "val_adba", "selected_epoch": 2}),
        encoding="utf-8",
    )
    (checkpoint_dir / "last.pth").write_bytes(b"last weights")

    manifest = build_cleanup_manifest(
        project_root=tmp_path,
        scan_roots=[outputs],
        include_resources=False,
        now=NOW,
    )

    candidates = _records_by_rule(manifest, "candidates")
    protected = _records_by_rule(manifest, "protected")
    assert "output.ambiguous_other" in candidates
    assert "checkpoint.last_recoverable" in candidates
    assert "checkpoint.reproducible_protected" in protected
    assert candidates["checkpoint.last_recoverable"][0]["path"].endswith("checkpoints/last.pth")
    run_candidate = candidates["output.ambiguous_other"][0]
    assert run_candidate["run_summary"]["checkpoint_count"] == 2
    assert run_candidate["run_summary"]["checkpoint_total_size_bytes"] > 0
    assert manifest["summary"]["candidate_count"] >= 2
    assert manifest["summary"]["candidate_total_size_bytes"] > 0


def test_cleanup_manifest_protects_tracked_files_and_protected_roots(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    debug_dir = tmp_path / "docs" / "_debug"
    debug_dir.mkdir(parents=True)
    tracked_file = debug_dir / "tracked.txt"
    tracked_file.write_text("tracked", encoding="utf-8")
    subprocess.run(["git", "add", "docs/_debug/tracked.txt"], cwd=tmp_path, check=True, capture_output=True)

    manifest = build_cleanup_manifest(
        project_root=tmp_path,
        scan_roots=[debug_dir],
        include_resources=False,
        now=NOW,
    )

    assert manifest["candidates"] == []
    protected = _records_by_rule(manifest, "protected")
    assert "transient.debug" in protected
    record = protected["transient.debug"][0]
    assert record["tracked"] is True
    assert "git_tracked" in record["protection_reasons"]
    assert "protected_root:docs" in record["protection_reasons"]


def test_cleanup_manifest_covers_caches_debug_plan_and_apply_requires_confirmation(tmp_path: Path):
    debug_dir = tmp_path / "outputs" / "_debug"
    debug_dir.mkdir(parents=True)
    (debug_dir / "scratch.txt").write_text("scratch", encoding="utf-8")
    plan_dir = tmp_path / "outputs" / "_plan_check"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.txt").write_text("plan", encoding="utf-8")
    pycache = tmp_path / "cache" / "pkg" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "module.cpython-311.pyc").write_bytes(b"pyc")
    pytest_cache = tmp_path / ".pytest_cache" / "v"
    pytest_cache.mkdir(parents=True)
    (pytest_cache / "cache").write_text("pytest", encoding="utf-8")
    backup = tmp_path / "outputs" / "personal_backup.zip"
    backup.write_bytes(b"zip")

    manifest = build_cleanup_manifest(
        project_root=tmp_path,
        scan_roots=[tmp_path / "outputs", tmp_path / "cache", tmp_path / ".pytest_cache"],
        include_resources=False,
        now=NOW,
    )
    rules = _records_by_rule(manifest, "candidates")
    assert "transient.debug" in rules
    assert "transient.plan_check" in rules
    assert "cache.python_bytecode" in rules
    assert "cache.pytest" in rules
    assert "archive.personal_backup" in rules

    manifest_path = write_cleanup_manifest(manifest, output_path=tmp_path / "cleanup_manifest.json")
    with pytest.raises(ValueError, match="Deletion refused"):
        apply_cleanup_manifest(manifest_path, project_root=tmp_path)

    report = apply_cleanup_manifest(
        manifest_path,
        project_root=tmp_path,
        confirm_delete=True,
        report_path=tmp_path / "delete_report.json",
    )
    assert report["summary"]["deleted_count"] >= 3
    assert not debug_dir.exists()
    assert not plan_dir.exists()
    assert not backup.exists()
    assert (tmp_path / "delete_report.json").exists()


def test_cleanup_apply_skips_candidate_containing_manifest_protected_path(tmp_path: Path):
    outputs = tmp_path / "outputs"
    candidate = outputs / "quick_smoke_run"
    checkpoint_dir = candidate / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best.pth"
    best.write_bytes(b"best")
    manifest = {
        "metadata": {
            "project_root": str(tmp_path),
            "scan_roots": [str(outputs)],
            "rules_version": "runtime-artifact-cleanup.v1",
        },
        "candidates": [
            {
                "path": str(candidate),
                "relative_path": "outputs/quick_smoke_run",
                "rule_id": "transient.smoke",
                "protected": False,
                "size_bytes": 4,
                "mtime": None,
            }
        ],
        "protected": [
            {
                "path": str(best),
                "relative_path": "outputs/quick_smoke_run/checkpoints/best.pth",
                "rule_id": "checkpoint.reproducible_protected",
                "protected": True,
                "action": "protect",
            }
        ],
    }
    manifest_path = tmp_path / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = apply_cleanup_manifest(
        manifest_path,
        project_root=tmp_path,
        confirm_delete=True,
        report_path=tmp_path / "delete_report.json",
    )

    assert candidate.exists()
    assert best.exists()
    assert report["summary"]["deleted_count"] == 0
    assert report["summary"]["skipped_count"] == 1
    assert report["skipped"][0]["reason"] == "manifest_protected_path_overlap"


def test_cleanup_manifest_classifies_retired_hist_outputs_without_hist2_false_positive(tmp_path: Path):
    outputs = tmp_path / "outputs"
    for name in (
        "hist_beam_loso",
        "history_anchor_20ep_gpu0",
        "image_only_legal_seed0",
        "p3_v8_a2a5_single_target_seed0_4x3090",
        "v9_fast_4gpu",
        "_debug_iofix_plan",
        "gps_window_baseline_plan_check",
        "quick_smoke_run",
    ):
        directory = outputs / name
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text(name, encoding="utf-8")
    hist2 = outputs / "gps_window_baseline_target_calibrated_hist2"
    hist2.mkdir(parents=True)
    (hist2 / "metrics.json").write_text("{}", encoding="utf-8")
    for partition in ("analysis", "cache", "features", "training"):
        directory = outputs / partition / "current_mainline"
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text(partition, encoding="utf-8")

    manifest = build_cleanup_manifest(
        project_root=tmp_path,
        scan_roots=[outputs],
        include_resources=False,
        now=NOW,
    )

    candidates = _records_by_rule(manifest, "candidates")
    protected = _records_by_rule(manifest, "protected")
    assert "retired.hist_output" in candidates
    assert "retired.history_anchor_hist_output" in candidates
    assert "retired.image_only_hist_output" in candidates
    assert "retired.p3_hist_probe" in candidates
    assert "retired.v8_v9_hist_probe" in candidates
    assert "transient.debug" in candidates
    assert "transient.plan_check" in candidates
    assert "transient.smoke" in candidates
    assert "protected.current_mainline_output" in protected
    hist2_rules = {
        rule_id
        for record in manifest["candidates"]
        if record["relative_path"] == "outputs/gps_window_baseline_target_calibrated_hist2"
        for rule_id in record.get("matched_rules", [record.get("rule_id")])
    }
    assert not {rule_id for rule_id in hist2_rules if rule_id.startswith("retired.")}


def test_cleanup_console_script_help_works_through_conda():
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda is not available")

    result = subprocess.run(
        [conda, "run", "-n", "kd_mm_beam", "kd-sensing-clean-runtime-artifacts", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--confirm-delete" in result.stdout
