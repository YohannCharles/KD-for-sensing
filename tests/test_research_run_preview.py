import datetime as dt
import json
from pathlib import Path

from kd_sensing.diagnostics.research_run_preview import (
    build_budget_manifest,
    build_research_run_preview,
    validate_evidence_preview,
)


NOW = dt.datetime(2026, 7, 5, 12, 0, tzinfo=dt.timezone.utc)


def _summary(tmp_path: Path) -> dict:
    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": "2026-07-05T12:00:00Z",
            "project_root": str(tmp_path),
            "candidate_only": True,
            "read_only": True,
            "does_not_update_claim_registry": True,
        },
        "active_changes": [{"name": "add-research-run-preview-loop", "status": "active"}],
        "run_state_counts": {"complete": 1},
        "resources": {"gpu_available": False, "gpu_count": 0, "process_count": 0, "memory": {}},
        "claim_counts": {"strict": 1},
        "paper_readiness": {
            "status": "candidate_only",
            "pending_or_unverified_count": 1,
            "candidate_only_count": 1,
            "missing_field_counts": {"checkpoint_provenance": 1},
            "upgradable_candidate_count": 0,
            "paper_export_gate": {
                "candidate_only_excluded": True,
                "main_table_hard_exclude_status_markers": ["pending", "unverified"],
            },
            "next_action_hints": ["keep candidate-only caveat"],
        },
        "candidates": [
            {
                "candidate_id": "candidate-one",
                "run_name": "proto_seed1",
                "method": "proto",
                "seed": 1,
                "comparability_status": "strict",
                "claim_status": "draft",
                "candidate_only": True,
                "artifact_paths": {"source": str(tmp_path / "summary.csv")},
                "next_action_hints": ["manual review"],
            }
        ],
        "upgradable_candidates": [],
        "next_action_hints": ["manual review"],
        "warnings": [],
    }


def test_preview_loop_reuses_dashboard_summary_and_writes_manifest_without_training_side_effects(tmp_path: Path):
    manifest = build_research_run_preview(
        project_root=tmp_path,
        outputs=[tmp_path / "outputs"],
        logs=None,
        output_dir=tmp_path / "preview",
        dashboard_summary=_summary(tmp_path),
        include_resources=False,
        budget={"workflow_id": "smoke-preview", "change_id": "add-research-run-preview-loop"},
        now=NOW,
    )

    assert manifest["metadata"]["does_not_start_training"] is True
    assert manifest["metadata"]["does_not_read_real_dataset"] is True
    assert manifest["metadata"]["does_not_load_checkpoint"] is True
    assert manifest["preview_qa"]["status"] == "pass"
    assert manifest["checks"][0]["status"] == "planned"
    assert "kd-sensing-research-preview" in manifest["run_recipe"]["smoke_dev"]["command"]
    assert "python -m kd_sensing.cli.research_preview" in manifest["run_recipe"]["smoke_dev"]["module_fallback"]
    assert Path(manifest["outputs"]["preview_manifest"]).exists()
    assert Path(manifest["outputs"]["dashboard_html"]).read_text(encoding="utf-8").startswith("<!doctype html>")


def test_preview_qa_catches_empty_missing_columns_candidate_pending_remote_and_escaping(tmp_path: Path):
    good_table = tmp_path / "good.csv"
    good_table.write_text(
        "method,claim_status,candidate_only,comparability_status,provenance,caveat,metric,value\n"
        "proto,draft,true,strict,summary.csv,candidate-only; manual review,top1,0.5\n"
        "baseline,pending,false,not_comparable,summary.csv,pending seed evidence,top1,0.4\n",
        encoding="utf-8",
    )
    empty_table = tmp_path / "empty.csv"
    empty_table.write_text("method,claim_status,caveat,comparability_status,provenance\n", encoding="utf-8")
    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("method,claim_status\nproto,reviewed\n", encoding="utf-8")
    bad_candidate = tmp_path / "bad_candidate.csv"
    bad_candidate.write_text(
        "method,claim_status,candidate_only,comparability_status,provenance,caveat\n"
        "proto,reviewed,true,strict,summary.csv,\n",
        encoding="utf-8",
    )
    remote_html = tmp_path / "remote.html"
    remote_html.write_text(
        "<!doctype html><html><body><h2>Metadata</h2><h2>Claim Readiness</h2><h2>Paper Readiness</h2>"
        "<script src=\"https://cdn.example/app.js\"></script>candidate caveat</body></html>",
        encoding="utf-8",
    )
    escaped_html = tmp_path / "escaped.html"
    escaped_html.write_text(
        "<!doctype html><html><body><h2>Metadata</h2><h2>Claim Readiness</h2><h2>Paper Readiness</h2>"
        "candidate-only &lt;script&gt; escaped</body></html>",
        encoding="utf-8",
    )

    qa = validate_evidence_preview(
        {
            "table": [good_table, empty_table, missing_columns, bad_candidate],
            "html": [remote_html, escaped_html],
        }
    )

    messages = [issue["message"] for issue in qa["issues"]]
    assert qa["status"] == "fail"
    assert "table has no data rows" in messages
    assert "missing required field: caveat" in messages
    assert any("candidate-only row marked as reviewed" in message for message in messages)
    assert "remote dependency or script tag is not allowed" in messages
    assert all(issue["path"] != str(escaped_html) for issue in qa["issues"])


def test_preview_qa_checks_figure_checklist_and_conclusion_caveats(tmp_path: Path):
    figure = tmp_path / "figure.json"
    figure.write_text(json.dumps({"rows": [{"method": "proto", "metric": "top1", "caveat": "draft"}]}), encoding="utf-8")
    empty_figure = tmp_path / "empty_figure.json"
    empty_figure.write_text(json.dumps({"rows": []}), encoding="utf-8")
    checklist = tmp_path / "checklist.csv"
    checklist.write_text("item,claim_status,caveat\nseed,pending,\n", encoding="utf-8")
    conclusion = tmp_path / "conclusion.md"
    conclusion.write_text("Pending evidence remains incomplete.", encoding="utf-8")

    qa = validate_evidence_preview(
        {
            "figure_data": [figure, empty_figure],
            "checklist": [checklist],
            "conclusion": [conclusion],
        }
    )

    assert qa["status"] == "fail"
    messages = [issue["message"] for issue in qa["issues"]]
    assert "figure data has no rows" in messages
    assert "pending checklist item missing caveat at row 2" in messages
    assert "pending/incomplete conclusion must include caveat" in messages


def test_budget_manifest_reports_missing_long_run_fields_and_source_boundary():
    manifest = build_budget_manifest(
        {
            "workflow_id": "full-sweep",
            "change_id": "add-research-run-preview-loop",
            "long_run": True,
            "dataset_family": "deepsense6g",
            "reads_real_dataset": True,
            "output_root": "outputs/scenes31_34_main_lmdb",
            "stop_conditions": ["stop on first killed run"],
        }
    )

    missing = manifest["validation"]["missing_required_fields"]
    assert manifest["artifacts_not_committed"] is True
    assert "config_path (one required for long_run)" in missing
    assert "checkpoint_plan (long_run must declare write/read policy)" in missing
    assert "gpu (long_run must declare GPU/CPU plan)" in missing
