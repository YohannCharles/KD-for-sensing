import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import kd_sensing.diagnostics.run_index as run_index_module

from kd_sensing.config.io import dump_config
from kd_sensing.diagnostics.runtime_artifact_cleanup import (
    apply_cleanup_manifest,
    apply_runtime_output_organize_manifest,
    build_cleanup_manifest,
    build_runtime_output_organize_manifest,
    write_cleanup_manifest,
    write_runtime_output_organize_manifest,
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


def _init_git(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)


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
    assert candidates["checkpoint.last_recoverable"][0]["filesystem_type"] == "regular_file"
    run_candidate = candidates["output.ambiguous_other"][0]
    assert run_candidate["filesystem_type"] == "directory"
    assert run_candidate["run_summary"]["checkpoint_count"] == 2
    assert run_candidate["run_summary"]["checkpoint_total_size_bytes"] > 0
    assert manifest["summary"]["candidate_count"] >= 2
    assert manifest["summary"]["candidate_total_size_bytes"] > 0


def test_cleanup_manifest_protects_old_run_with_matching_live_process(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "other", "live_old_run")
    old = (NOW - dt.timedelta(days=3)).timestamp()
    for path in [run_dir, *run_dir.rglob("*")]:
        os.utime(path, (old, old))
    processes = [
        {
            "pid": 1234,
            "cmdline": f"python -m kd_sensing.cli.train output.run_name={run_dir.name}",
            "rss_mb": 256.0,
            "run_name": run_dir.name,
            "gpu_indices": [0],
        }
    ]
    monkeypatch.setattr(run_index_module, "collect_python_processes", lambda: processes)
    monkeypatch.setattr(run_index_module, "collect_resource_snapshot", lambda records: {"processes": list(records)})

    manifest = build_cleanup_manifest(
        project_root=tmp_path,
        scan_roots=[outputs],
        include_resources=True,
        now=NOW,
    )

    record = next(item for item in manifest["protected"] if Path(item["path"]) == run_dir)
    assert record["run_summary"]["state"] == "running"
    assert "run_state_running" in record["protection_reasons"]


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
    _init_git(tmp_path)
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
            "schema_version": 1,
            "project_root": str(tmp_path),
            "scan_roots": [str(outputs)],
            "allowed_roots": [str(outputs)],
            "rules_version": "runtime-artifact-cleanup.v2",
        },
        "candidates": [
            {
                "path": str(candidate),
                "relative_path": "outputs/quick_smoke_run",
                "rule_id": "transient.smoke",
                "protected": False,
                "filesystem_type": "directory",
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
    _init_git(tmp_path)

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


def test_cleanup_apply_rejects_empty_path_before_deleting_any_candidate(tmp_path: Path):
    _init_git(tmp_path)
    outputs = tmp_path / "outputs"
    candidate = outputs / "_debug"
    candidate.mkdir(parents=True)
    (candidate / "keep.txt").write_text("keep", encoding="utf-8")
    manifest = build_cleanup_manifest(project_root=tmp_path, scan_roots=[outputs], include_resources=False, now=NOW)
    manifest["candidates"].append({"path": "", "size_bytes": 0, "mtime": None})
    manifest_path = tmp_path / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="empty path"):
        apply_cleanup_manifest(manifest_path, project_root=tmp_path, confirm_delete=True)

    assert candidate.exists()


def test_cleanup_apply_refuses_scan_root_and_project_root(tmp_path: Path):
    _init_git(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifest = build_cleanup_manifest(project_root=tmp_path, scan_roots=[outputs], include_resources=False, now=NOW)
    manifest["candidates"] = [
        {
            "path": str(outputs),
            "filesystem_type": "directory",
            "size_bytes": 0,
            "mtime": None,
            "rule_id": "unsafe.scan_root",
        },
    ]
    manifest_path = tmp_path / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = apply_cleanup_manifest(manifest_path, project_root=tmp_path, confirm_delete=True)

    assert outputs.exists()
    assert report["skipped"][0]["reason"] == "protected_root_candidate"

    manifest["metadata"]["scan_roots"] = [str(tmp_path)]
    manifest["metadata"]["allowed_roots"] = [str(tmp_path)]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe root"):
        apply_cleanup_manifest(manifest_path, project_root=tmp_path, confirm_delete=True)


def test_cleanup_apply_rejects_filesystem_type_drift_before_deletion(tmp_path: Path):
    _init_git(tmp_path)
    outputs = tmp_path / "outputs"
    candidate = outputs / "_debug"
    candidate.mkdir(parents=True)
    (candidate / "old.txt").write_text("old", encoding="utf-8")
    manifest = build_cleanup_manifest(project_root=tmp_path, scan_roots=[outputs], include_resources=False, now=NOW)
    record = next(item for item in manifest["candidates"] if Path(item["path"]) == candidate)
    assert record["filesystem_type"] == "directory"

    shutil.rmtree(candidate)
    candidate.write_text("replacement", encoding="utf-8")
    manifest_path = tmp_path / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = apply_cleanup_manifest(manifest_path, project_root=tmp_path, confirm_delete=True)

    assert candidate.is_file()
    skipped = next(item for item in report["skipped"] if Path(item["path"]) == candidate)
    assert skipped["reason"] == "filesystem_type_changed_since_manifest"
    assert skipped["manifest_filesystem_type"] == "directory"
    assert skipped["current_filesystem_type"] == "regular_file"


def test_cleanup_apply_fails_closed_when_git_state_is_unavailable(tmp_path: Path, monkeypatch):
    _init_git(tmp_path)
    outputs = tmp_path / "outputs"
    candidate = outputs / "_debug"
    candidate.mkdir(parents=True)
    (candidate / "keep.txt").write_text("keep", encoding="utf-8")
    manifest = build_cleanup_manifest(project_root=tmp_path, scan_roots=[outputs], include_resources=False, now=NOW)
    manifest_path = tmp_path / "cleanup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "kd_sensing.diagnostics.runtime_artifact_cleanup_apply.collect_git_tracked_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("git unavailable")),
    )
    with pytest.raises(RuntimeError, match="git unavailable"):
        apply_cleanup_manifest(manifest_path, project_root=tmp_path, confirm_delete=True)

    assert candidate.exists()


def test_cleanup_manifest_covers_transient_outputs_and_current_partition_protection(tmp_path: Path):
    outputs = tmp_path / "outputs"
    for name in (
        "_debug_iofix_plan",
        "gps_window_baseline_plan_check",
        "quick_smoke_run",
    ):
        directory = outputs / name
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_text(name, encoding="utf-8")
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
    assert "transient.debug" in candidates
    assert "transient.plan_check" in candidates
    assert "transient.smoke" in candidates
    assert "protected.current_mainline_output" in protected
    assert not {rule_id for rule_id in candidates if rule_id.startswith("retired.")}


def test_runtime_output_organize_manifest_classifies_legacy_outputs_and_protects_cache(tmp_path: Path):
    outputs = tmp_path / "outputs"
    root_run = _write_started_run(outputs, "root_run")
    deepsense_cfg = _base_cfg("root_run")
    deepsense_cfg["data"]["dataset"] = {"type": "deepsense6g"}
    dump_config(deepsense_cfg, root_run / "final_config.yaml")
    dump_config(deepsense_cfg, root_run / "resolved_config.yaml")
    numeric = outputs / "31"
    numeric.mkdir(parents=True)
    (numeric / "legacy.txt").write_text("legacy", encoding="utf-8")
    registry = outputs / "best_checkpoints"
    registry.mkdir(parents=True)
    (registry / "model.pth").write_bytes(b"weights")
    legacy_eval = outputs / "eval_old"
    legacy_eval.mkdir(parents=True)
    (legacy_eval / "metrics.json").write_text("{}", encoding="utf-8")
    cache = outputs / "cache"
    cache.mkdir(parents=True)
    (cache / "summary.json").write_text("{}", encoding="utf-8")

    manifest = build_runtime_output_organize_manifest(project_root=tmp_path, outputs_root=outputs, now=NOW)
    by_rule = {record["rule_id"]: record for record in manifest["plans"]}

    assert by_rule["organize.legacy_root_run"]["action"] == "move"
    assert by_rule["organize.legacy_root_run"]["target_path"].endswith("outputs/scene31/root_run")
    assert by_rule["organize.legacy_numeric_scene"]["action"] == "archive"
    assert by_rule["organize.legacy_registry"]["action"] == "review"
    assert by_rule["organize.legacy_registry"]["requires_manual_review"] is True
    assert by_rule["organize.legacy_evaluation"]["target_path"].endswith("outputs/archive/legacy_eval_runs/eval_old")
    assert by_rule["organize.cache_protected"]["action"] == "protect"
    assert by_rule["organize.cache_protected"]["protected"] is True
    assert root_run.exists()
    assert numeric.exists()
    assert registry.exists()
    assert legacy_eval.exists()
    assert cache.exists()


def test_runtime_output_organize_apply_requires_confirmation_and_skips_conflicts(tmp_path: Path):
    outputs = tmp_path / "outputs"
    root_run = _write_started_run(outputs, "root_run")
    deepsense_cfg = _base_cfg("root_run")
    deepsense_cfg["data"]["dataset"] = {"type": "deepsense6g"}
    dump_config(deepsense_cfg, root_run / "final_config.yaml")
    dump_config(deepsense_cfg, root_run / "resolved_config.yaml")
    target = outputs / "scene31" / "root_run"
    target.mkdir(parents=True)
    (target / "final_config.yaml").write_text("existing", encoding="utf-8")

    manifest = build_runtime_output_organize_manifest(project_root=tmp_path, outputs_root=outputs, now=NOW)
    manifest_path = write_runtime_output_organize_manifest(manifest, output_path=tmp_path / "organize.json")

    with pytest.raises(ValueError, match="Organization refused"):
        apply_runtime_output_organize_manifest(manifest_path, project_root=tmp_path)

    report = apply_runtime_output_organize_manifest(
        manifest_path,
        project_root=tmp_path,
        confirm_organize=True,
        report_path=tmp_path / "organize_report.json",
    )

    assert root_run.exists()
    assert target.exists()
    assert report["summary"]["moved_count"] == 0
    assert report["summary"]["skipped_count"] >= 1
    assert any(item["reason"] in {"manual_review_required", "target_exists"} for item in report["skipped"])


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


def test_organize_console_script_help_works_through_conda():
    conda = shutil.which("conda")
    if conda is None:
        pytest.skip("conda is not available")

    result = subprocess.run(
        [conda, "run", "-n", "kd_mm_beam", "python", "-m", "kd_sensing.cli.organize_runtime_outputs", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--confirm-organize" in result.stdout
