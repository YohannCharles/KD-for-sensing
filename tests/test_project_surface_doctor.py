import json
import subprocess
from pathlib import Path

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

    assert parsed["sections"]["configs"]["tracked_count"] == report["sections"]["configs"]["tracked_count"]
    assert "# Project Surface Doctor" in markdown
    assert "## Configs" in markdown


def _git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()
