import json
import subprocess
from pathlib import Path

import kd_sensing.diagnostics.project_surface_doctor as doctor
from kd_sensing.cli.project_surface_doctor import build_parser as build_project_surface_doctor_parser
from kd_sensing.diagnostics.project_surface_doctor import (
    build_project_surface_report,
    doctor_should_fail,
    render_project_surface_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_project_surface_doctor_reports_tracked_sections_without_writes():
    before = _git_status_short()
    report = build_project_surface_report(ROOT, fail_on="none")
    after = _git_status_short()

    assert before == after
    assert report["metadata"]["read_only"] is True
    assert "dataset/" in report["metadata"]["scan_policy"]["excluded_roots"]
    assert report["sections"]["scripts"]["tracked_count"] > 0
    assert report["sections"]["configs"]["tracked_count"] > 0
    assert report["sections"]["configs"]["virtual_routes"]
    assert report["sections"]["hotspots"]["registered_count"] > 0
    assert not doctor_should_fail(report)

    for issue in report["issues"]:
        assert issue["path"]
        assert issue["source"]
        assert issue["validation"].startswith(("conda run -n kd_mm_beam", "openspec "))


def test_project_surface_doctor_json_and_markdown_rendering():
    report = build_project_surface_report(ROOT, scopes=("configs",), fail_on="none")

    parsed = json.loads(render_project_surface_report(report, format="json"))
    markdown = render_project_surface_report(report, format="markdown")
    full_json = json.loads(render_project_surface_report(report, format="json", dump_inventory=True))
    full_markdown = render_project_surface_report(report, format="markdown", dump_inventory=True)

    assert "sections" not in parsed
    assert parsed["inventory_omitted"] is True
    assert "# Project Surface Doctor" in markdown
    assert "## Configs" not in markdown
    assert "## Next Action" in markdown
    assert full_json["sections"]["configs"]["tracked_count"] == report["sections"]["configs"]["tracked_count"]
    assert full_json["inventory_omitted"] is False
    assert "## Configs" in full_markdown


def test_project_surface_doctor_cli_requires_explicit_inventory_dump():
    parser = build_project_surface_doctor_parser()
    default_args = parser.parse_args([])
    dump_args = parser.parse_args(["--dump-inventory"])

    assert default_args.dump_inventory is False
    assert dump_args.dump_inventory is True


def test_project_surface_doctor_classifies_shrunk_experiment_config_families():
    report = build_project_surface_report(ROOT, scopes=("configs",), fail_on="none")
    configs = report["sections"]["configs"]
    entries = {entry["path"]: entry for entry in configs["entries"]}

    assert entries["configs/fusion/u_mask_beam_jepa_smoke.yaml"]["family"] == "canonical fusion"
    assert entries["configs/fusion/physics_informed_mmw_debug.yaml"]["lifecycle"] == "current"
    assert entries["configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml"]["lifecycle"] == "current"
    assert entries["configs/scene31/templates/main_v3_proto_es20_base.yaml"]["lifecycle"] == "generated/recipe-backed"
    assert configs["recipe_migration_candidates"]
    assert any(
        "configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml" in candidate["paths"]
        for candidate in configs["recipe_migration_candidates"]
    )


def test_project_surface_doctor_cli_surface_scope_checks_public_entrypoints():
    report = build_project_surface_report(ROOT, scopes=("cli-surface",), fail_on="none")
    cli_surface = report["sections"]["cli_surface"]

    assert cli_surface["public_count"] == cli_surface["classified_count"]
    assert cli_surface["public_count"] == cli_surface["help_smoke_count"]
    assert cli_surface["stale_references"] == []
    assert not doctor_should_fail(report)
    assert all(entry["help_smoke"] for entry in cli_surface["entries"])


def test_project_surface_doctor_security_scope_flags_guardrail_risks(monkeypatch):
    original_git_ls_files = doctor._git_ls_files
    original_read_text = doctor._read_text
    protected_env = "/root/" + ".container_env"
    fake_cli = "kd-sensing-" + "train"
    fake_token = "ghp_" + "123456789012345678901234567890123456"
    credential_field = "PASS" + "WD"
    fake_text = {
        "scripts/bad_runner.sh": "\n".join(
            [
                f'echo "{credential_field}={fake_cli} --config configs/image/strong.yaml" >> {protected_env}',
                "rm -rf outputs/bad-run",
                f"{fake_cli} --config configs/image/strong.yaml",
            ]
        ),
        "configs/leaked.yaml": f'token: "{fake_token}"',
    }
    fake_artifacts = [
        "dataset/raw/sample.bin",
        "outputs/run/checkpoints/best.ckpt",
        "logs/train.log",
        "cache/tmp.pkl",
    ]

    def fake_git_ls_files(root: Path) -> list[str]:
        return [*original_git_ls_files(root), *fake_text, *fake_artifacts]

    def fake_read_text(path: Path) -> str:
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path in fake_text:
            return fake_text[rel_path]
        return original_read_text(path)

    monkeypatch.setattr(doctor, "_git_ls_files", fake_git_ls_files)
    monkeypatch.setattr(doctor, "_read_text", fake_read_text)

    report = build_project_surface_report(ROOT, scopes=("security",), fail_on="none")
    kinds = {issue["kind"] for issue in report["issues"]}
    artifact_paths = {item["path"] for item in report["sections"]["security"]["runtime_artifact_hits"]}

    assert "runtime_artifact_tracked" in kinds
    assert "secret_literal" in kinds
    assert "protected_system_config_mutation" in kinds
    assert "dangerous_shell_runner_command" in kinds
    assert "dataset/raw/sample.bin" in artifact_paths
    assert "outputs/run/checkpoints/best.ckpt" in artifact_paths
    assert report["sections"]["security"]["policy"]["allows_manifest_cleanup_confirmation"] is True


def test_project_surface_doctor_closeout_scope_classifies_dirty_state(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_git_status_short",
        lambda root: [
            " M README.md",
            "?? docs/local_note.md",
            "?? outputs/tmp/report.json",
            " D openspec/changes/old-change/tasks.md",
            "?? openspec/changes/archive/2026-07-05-old-change/",
        ],
    )
    monkeypatch.setattr(
        doctor,
        "_openspec_list",
        lambda root: {
            "available": True,
            "changes": [{"name": "old-change", "completedTasks": 2, "totalTasks": 2, "status": "in-progress"}],
        },
    )

    report = build_project_surface_report(ROOT, scopes=("closeout",), fail_on="none")
    closeout = report["sections"]["closeout"]
    kinds = {issue["kind"] for issue in report["issues"]}

    assert "complete_change_unarchived" in kinds
    assert "untracked_archive_change" in kinds
    assert "active_delete_archive_pair" in kinds
    assert closeout["worktree"]["category_counts"]["runtime_artifact"] == 1
    assert closeout["worktree"]["category_counts"]["openspec"] == 2
    assert closeout["policy"]["does_not_reset"] is True
    assert closeout["policy"]["does_not_archive"] is True


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()
