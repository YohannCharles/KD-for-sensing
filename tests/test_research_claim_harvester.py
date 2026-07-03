import datetime as dt
import json
from pathlib import Path

from kd_sensing.config.io import dump_config
from kd_sensing.diagnostics.research_claim_harvester import (
    apply_strict_comparability_gate,
    build_dashboard_summary,
    harvest_research_claims,
    ledger_records_from_candidates,
    read_training_run_artifact,
    training_run_claim_candidate,
    write_jsonl_ledger,
    write_ledger_csv,
)


NOW = dt.datetime(2026, 6, 1, 9, 0, tzinfo=dt.timezone.utc)


def _strict_cfg(name: str, *, seed: int = 7) -> dict:
    return {
        "experiment": {"name": name, "task": "fusion", "objective": "beam", "seed": seed},
        "data": {
            "dataset": {
                "type": "deepsense6g",
                "split": "test",
                "sample_count": 16,
                "label_space": "beam64",
                "beam_target_source": "current",
                "difficulty_digest": "difficulty-a",
            }
        },
        "output": {"run_name": name},
        "runtime": {
            "scene_scope": "scene31",
            "metric_profile": "scene31_missing",
            "target_source": "current",
            "run_family": "proto",
        },
    }


def _write_complete_run(root: Path, name: str = "proto_seed7") -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    dump_config(_strict_cfg(name), run_dir / "final_config.yaml")
    (run_dir / "metrics.json").write_text(json.dumps({"val_adba": 0.72, "top1": 0.61}), encoding="utf-8")
    (run_dir / "train_log.json").write_text(json.dumps({"epoch_logs": [{"epoch": 2}]}), encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "primary_metric": {"name": "val_adba", "value": 0.72},
                "best_checkpoint": str(run_dir / "checkpoints" / "best.pth"),
            }
        ),
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pth").write_bytes(b"weights")
    (checkpoint_dir / "best.pth.json").write_text(
        json.dumps({"selection_metric": "val_adba", "selected_epoch": 2}),
        encoding="utf-8",
    )
    return run_dir


def test_scene31_missing_pattern_reader_builds_strict_candidates(tmp_path: Path):
    eval_dir = tmp_path / "outputs" / "scene31" / "eval"
    eval_dir.mkdir(parents=True)
    artifact = eval_dir / "proto_missing_patterns.csv"
    artifact.write_text(
        "run_name,method,run_family,seed,pattern,split,sample_count,label_space,metric_profile,target_source,difficulty_digest,top1,dba\n"
        "proto_seed1,proto,proto,1,missing_gps,test,16,beam64,scene31_missing,current,difficulty-a,0.50,0.70\n"
        "proto_seed2,proto,proto,2,missing_gps,test,16,beam64,scene31_missing,current,difficulty-a,0.52,0.72\n",
        encoding="utf-8",
    )

    harvest = harvest_research_claims(eval_dir, run_index={"runs": [], "warnings": []}, now=NOW)

    assert len(harvest["candidates"]) == 2
    assert {candidate["comparability_status"] for candidate in harvest["candidates"]} == {"strict"}
    first = harvest["candidates"][0]
    assert first["candidate_only"] is True
    assert first["claim_status"] == "draft"
    assert first["run_name"] == "proto_seed1"
    assert first["metrics"]["top1"] == 0.5
    assert first["artifact_paths"]["source"].endswith("proto_missing_patterns.csv")


def test_training_run_reader_extracts_provenance_and_writes_ledgers(tmp_path: Path):
    run_dir = _write_complete_run(tmp_path / "outputs" / "scene31")

    record = read_training_run_artifact(run_dir)
    candidate = training_run_claim_candidate(run_dir, generated_at="2026-06-01T09:00:00Z")
    gated = apply_strict_comparability_gate([candidate])
    records = ledger_records_from_candidates(gated, generated_at="2026-06-01T09:00:00Z")
    jsonl_path = write_jsonl_ledger(records, ledger_dir=tmp_path / "ledger", now=NOW)
    csv_path = write_ledger_csv(records, output_path=tmp_path / "ledger.csv")

    assert record["config"]["config_digest"]
    assert record["checkpoint_provenance"]["status"] == "complete"
    assert gated[0]["comparability_status"] == "strict"
    assert records[0]["artifact_paths"]["checkpoint"].endswith("checkpoints/best.pth")
    assert json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])["claim_status"] == "draft"
    assert "comparability_status" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_gate_marks_conflicts_and_incomplete_provenance_for_review(tmp_path: Path):
    base = {
        "candidate_id": "c1",
        "run_id": "r1",
        "run_name": "a",
        "method": "proto",
        "pattern": "missing_gps",
        "seed": 1,
        "split": "test",
        "sample_count": 16,
        "label_space": "beam64",
        "metric_profile": "scene31_missing",
        "target_source": "current",
        "difficulty_digest": "difficulty-a",
        "run_family": "proto",
        "metrics": {"top1": 0.5},
        "artifact_paths": {},
        "claim_status": "draft",
        "candidate_only": True,
        "warnings": [],
    }
    conflict = {**base, "candidate_id": "c2", "run_id": "r2", "seed": 2, "sample_count": 20}

    gated = apply_strict_comparability_gate([base, conflict])

    assert {candidate["comparability_status"] for candidate in gated} == {"not_comparable"}
    assert any(warning["field"] == "sample_count" for warning in gated[1]["warnings"])

    incomplete = tmp_path / "outputs" / "scene31" / "no_sidecar"
    incomplete.mkdir(parents=True)
    dump_config(_strict_cfg("no_sidecar"), incomplete / "final_config.yaml")
    (incomplete / "metrics.json").write_text(json.dumps({"val_adba": 0.7}), encoding="utf-8")
    (incomplete / "checkpoints").mkdir()
    (incomplete / "checkpoints" / "best.pth").write_bytes(b"weights")

    candidate = training_run_claim_candidate(incomplete, generated_at="2026-06-01T09:00:00Z")
    gated_incomplete = apply_strict_comparability_gate([candidate])[0]

    assert gated_incomplete["comparability_status"] == "needs_review"
    assert any(warning["field"] == "checkpoint_provenance" for warning in gated_incomplete["warnings"])


def test_dashboard_aggregates_active_change_resources_and_next_actions(tmp_path: Path):
    run_dir = _write_complete_run(tmp_path / "outputs" / "scene31")
    run_index = {
        "runs": [
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "state": "running",
                "claim_harvester": {"run_dir": str(run_dir), "run_name": run_dir.name},
            }
        ],
        "resources": {
            "gpus": {"available": True, "devices": [{"index": 0}], "processes": []},
            "processes": [{"pid": 123}],
            "memory": {"available": True, "total_mb": 1024},
        },
        "warnings": [],
    }

    summary = build_dashboard_summary(
        project_root=tmp_path,
        outputs=tmp_path / "outputs",
        logs=None,
        run_index=run_index,
        active_changes=[{"name": "add-research-claim-harvester-dashboard", "status": "active"}],
        now=NOW,
    )

    assert summary["metadata"]["read_only"] is True
    assert summary["metadata"]["does_not_update_claim_registry"] is True
    assert summary["active_changes"][0]["name"] == "add-research-claim-harvester-dashboard"
    assert summary["run_state_counts"] == {"running": 1}
    assert summary["resources"]["gpu_count"] == 1
    assert summary["claim_counts"] == {"strict": 1}
    assert summary["candidates"][0]["candidate_only"] is True
