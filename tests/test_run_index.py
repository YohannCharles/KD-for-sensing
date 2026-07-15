import datetime as dt
import json
import os
from pathlib import Path

from kd_sensing.config.io import dump_config
from kd_sensing.diagnostics.run_index import (
    RunIndexFilters,
    build_run_card,
    build_run_index,
    parse_failure_patterns,
    render_run_csv,
    render_run_table,
    write_run_card,
)
from kd_sensing.diagnostics.run_index_resources import collect_resource_snapshot, redact_command
from kd_sensing.engine.run_status import (
    write_complete_status,
    write_failed_status_for_active_run,
    write_running_status,
)


NOW = dt.datetime(2026, 5, 24, 12, 0, tzinfo=dt.timezone.utc)


def _base_cfg(name: str) -> dict:
    return {
        "experiment": {"name": name, "task": "fusion", "objective": "beam", "seed": 7},
        "data": {"dataset": {"type": "deepsense6g"}},
        "model": {"primary": {"modalities": ["image", "lidar"]}},
        "output": {"run_name": name},
        "runtime": {"prediction_objective": {"name": "beam", "primary_metric": "val_adba"}},
    }


def _write_started_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    cfg = _base_cfg(name)
    dump_config(cfg, run_dir / "final_config.yaml")
    dump_config(cfg, run_dir / "resolved_config.yaml")
    (run_dir / "startup_summary.json").write_text(json.dumps({"device": "cpu"}), encoding="utf-8")
    return run_dir


def _touch_old(path: Path, *, hours: int) -> None:
    timestamp = (NOW - dt.timedelta(hours=hours)).timestamp()
    for item in [path, *path.rglob("*")]:
        os.utime(item, (timestamp, timestamp))


def test_run_index_classifies_complete_run_and_extracts_summary(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "complete_run")
    (run_dir / "metrics.json").write_text(
        json.dumps({"objective": {"primary_metric": "val_adba"}, "val_adba": 0.42, "topk": {"1": 0.5}}),
        encoding="utf-8",
    )
    (run_dir / "train_log.json").write_text(json.dumps({"epoch_logs": [{"epoch": 1}]}), encoding="utf-8")
    (run_dir / "training_outputs.npz").write_bytes(b"placeholder")
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "evaluation_split": "test",
                "top1": 0.51,
                "selected_checkpoint": {"path": "checkpoints/last.pth", "checkpoint_role": "last"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pth").write_bytes(b"weights")
    (checkpoint_dir / "best.pth.json").write_text(json.dumps({"selection_metric": "val_adba"}), encoding="utf-8")
    (checkpoint_dir / "last.pth").write_bytes(b"recoverable")
    (run_dir / "tensorboard").mkdir()
    (run_dir / "tensorboard" / "events.out.tfevents.test").write_text("", encoding="utf-8")

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)

    assert set(index) == {"generated_at", "roots", "filters", "runs", "resources", "warnings"}
    assert len(index["runs"]) == 1
    run = index["runs"][0]
    assert run["state"] == "complete"
    assert run["runtime_layout"]["canonical_partition"] == "scene"
    assert run["runtime_layout"]["scope_slug"] == "scene31"
    assert run["config"]["dataset_family"] == "deepsense6g"
    assert run["config"]["config_digest"]
    assert run["config"]["modalities"] == ["image", "lidar"]
    assert run["metrics"]["primary"] == {"name": "val_adba", "value": 0.42}
    assert run["checkpoints"]["best_checkpoint"].endswith("checkpoints/best.pth")
    assert run["provenance"]["config_digest"] == run["config"]["config_digest"]
    assert run["provenance"]["checkpoint_provenance"]["sidecar_path"].endswith("checkpoints/best.pth.json")
    assert run["size_bytes"] > 0
    assert run["checkpoints"]["count"] == 2
    assert run["checkpoints"]["total_size_bytes"] >= len(b"weights") + len(b"recoverable")
    assert run["checkpoints"]["primary_checkpoint"].endswith("checkpoints/best.pth")
    retention = {item["name"]: item for item in run["checkpoints"]["retention"]["items"]}
    assert retention["best.pth"]["registry_protected"] is True
    assert retention["best.pth"]["selection_metadata"]["available"] is True
    assert retention["last.pth"]["registry_default_candidate"] is True
    assert run["tensorboard"]["event_count"] == 1
    assert run["cleanup"]["protected"] is False


def test_run_index_discovers_metrics_csv_and_eval_artifact_refs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "csv_run")
    (run_dir / "metrics.csv").write_text("epoch,val_adba\n1,0.33\n2,0.44\n", encoding="utf-8")
    (run_dir / "train_log.json").write_text(json.dumps({"epoch_logs": [{"epoch": 2}]}), encoding="utf-8")
    (run_dir / "training_outputs.npz").write_bytes(b"placeholder")
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "evaluation_split": "test",
                "top1": 0.51,
                "selected_checkpoint": {"path": "checkpoints/last.pth", "checkpoint_role": "last"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "csv_run_missing_patterns.csv").write_text(
        "run_name,method,seed,pattern,split,sample_count,label_space,metric_profile,target_source,difficulty_digest,top1\n"
        "csv_run,proto,7,missing_gps,test,16,beam64,scene31_missing,current,digest-a,0.5\n",
        encoding="utf-8",
    )

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)
    run = index["runs"][0]

    assert run["state"] == "complete"
    assert run["metrics"]["path"].endswith("metrics.csv")
    assert run["metrics"]["primary"] == {"name": "val_adba", "value": 0.44}
    assert run["metrics"]["final_test"]["evaluation_split"] == "test"
    assert run["metrics"]["final_test"]["primary"] == {"name": "top1", "value": 0.51}
    assert run["provenance"]["eval_artifacts"][0]["path"].endswith("csv_run_missing_patterns.csv")


def test_run_index_skips_non_run_partitions_by_default_but_allows_explicit_scan(tmp_path: Path):
    outputs = tmp_path / "outputs"
    cache_run = _write_started_run(outputs / "cache", "cached_run")
    active_run = _write_started_run(outputs / "scene31", "active_run")
    legacy_root_run = _write_started_run(outputs, "legacy_root_run")
    archive_run = _write_started_run(outputs / "archive", "archived_run")
    manifests_run = _write_started_run(outputs / "cleanup_manifests", "manifest_run")

    default_index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)
    explicit_cache = build_run_index(outputs=outputs / "cache", logs=None, include_resources=False, now=NOW)
    legacy_index = build_run_index(
        outputs=outputs,
        logs=None,
        include_resources=False,
        include_legacy_containers=True,
        now=NOW,
    )

    assert [run["run_name"] for run in default_index["runs"]] == [active_run.name]
    assert any("outputs/cache" in warning for warning in default_index["warnings"])
    assert any("outputs/archive" in warning for warning in default_index["warnings"])
    assert any("outputs/cleanup_manifests" in warning for warning in default_index["warnings"])
    assert any("legacy_root_run" in warning for warning in default_index["warnings"])
    assert [run["run_name"] for run in explicit_cache["runs"]] == [cache_run.name]
    assert {run["run_name"] for run in legacy_index["runs"]} == {active_run.name, legacy_root_run.name}
    assert explicit_cache["roots"]["explicit_non_run_partitions"] == [str((outputs / "cache").resolve())]
    assert archive_run.exists()
    assert manifests_run.exists()


def test_run_index_classifies_started_stale_partial_and_filters(tmp_path: Path):
    outputs = tmp_path / "outputs"
    scene = outputs / "scene31"
    fresh = _write_started_run(scene, "fresh_run")
    stale = _write_started_run(scene, "stale_run")
    partial = scene / "partial_run"
    partial.mkdir(parents=True)
    dump_config(_base_cfg("partial_run"), partial / "final_config.yaml")
    (partial / "metrics.json").write_text(json.dumps({"loss": 1.0}), encoding="utf-8")
    _touch_old(stale, hours=48)

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)
    states = {run["run_name"]: run["state"] for run in index["runs"]}

    assert states[fresh.name] == "started_no_metrics"
    assert states[stale.name] == "stale"
    assert states[partial.name] == "partial"

    filtered = build_run_index(
        outputs=outputs,
        logs=None,
        filters=RunIndexFilters(states=("stale",)),
        include_resources=False,
        now=NOW,
    )

    assert [run["run_name"] for run in filtered["runs"]] == ["stale_run"]


def test_run_index_uses_associated_killed_and_waiting_logs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    logs = tmp_path / "logs"
    logs.mkdir()
    scene = outputs / "scene31"
    killed_run = _write_started_run(scene, "killed_run")
    waiting_run = _write_started_run(scene, "waiting_run")
    (logs / "killed_run.log").write_text("epoch 1\nKilled\n", encoding="utf-8")
    (logs / "waiting_run.log").write_text(
        "waiting for checkpoint outputs/teacher/checkpoints/best.pth before launching\n",
        encoding="utf-8",
    )

    index = build_run_index(outputs=outputs, logs=logs, include_resources=False, now=NOW)
    runs = {run["run_name"]: run for run in index["runs"]}

    assert runs[killed_run.name]["state"] == "killed"
    assert runs[killed_run.name]["logs"][0]["failure"]["kind"] == "killed"
    assert runs[waiting_run.name]["state"] == "waiting"
    assert runs[waiting_run.name]["logs"][0]["failure"]["waiting_for"] == "outputs/teacher/checkpoints/best.pth"
    assert parse_failure_patterns("Traceback (most recent call last):\nValueError: bad")["kind"] == "traceback"
    assert parse_failure_patterns("ERROR conda.cli.main_run: execute failed")["kind"] == "conda_failed"


def test_run_index_marks_matching_process_as_running(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "live_run")
    processes = [
        {
            "pid": 1234,
            "cmdline": f"python -m kd_sensing.cli.train --config configs/live.yaml output.run_name={run_dir.name}",
            "rss_mb": 512.5,
            "run_name": run_dir.name,
            "gpu_indices": [0],
        }
    ]

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, processes=processes, now=NOW)
    run = index["runs"][0]

    assert run["state"] == "running"
    assert run["process"]["pid"] == 1234
    assert run["resources"]["process_rss_mb"] == 512.5
    assert run["resources"]["gpu_indices"] == [0]
    assert run["cleanup"]["protected"] is True
    assert "run_state_running" in run["cleanup"]["protection_reasons"]


def test_run_index_public_process_redacts_secrets_and_preserves_argv_boundaries(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "private_run")
    processes = [
        {
            "pid": 1234,
            "argv": [
                "python",
                "-m",
                "kd_sensing.cli.train",
                "--config",
                "configs/path with spaces.yaml",
                "--token",
                "top-secret",
                "service.password=hunter2",
                "--endpoint=https://user:pass@example.com/api",
                f"output.run_name={run_dir.name}",
            ],
            "rss_mb": 1.0,
            "run_name": run_dir.name,
            "gpu_indices": [],
        }
    ]

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, processes=processes, now=NOW)
    payload = json.dumps(index)
    public = index["runs"][0]["process"]

    assert "top-secret" not in payload
    assert "hunter2" not in payload
    assert "user:pass" not in payload
    assert public["argv"][4] == "configs/path with spaces.yaml"
    assert public["argv"][6] == "<redacted>"
    assert "service.password=<redacted>" in public["argv"]


def test_resource_snapshot_and_command_redaction_never_publish_raw_cmdline(monkeypatch):
    monkeypatch.setattr(
        "kd_sensing.diagnostics.run_index_resources.collect_gpu_snapshot",
        lambda: {"available": False, "devices": [], "processes": []},
    )
    monkeypatch.setattr(
        "kd_sensing.diagnostics.run_index_resources.collect_memory_snapshot",
        lambda: {"available": True},
    )
    snapshot = collect_resource_snapshot(
        [{"pid": 9, "cmdline": "python kd-sensing-train --api-key abc123 --config 'path with spaces.yaml'"}]
    )

    payload = json.dumps(snapshot)
    assert "abc123" not in payload
    assert snapshot["processes"][0]["argv"][-1] == "path with spaces.yaml"
    assert redact_command(["python", "--password", "secret"]) == "python --password '<redacted>'"


def test_supplied_resource_snapshot_is_resanitized_before_publication(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "private_resource_run")
    resources = {
        "memory": {"available": True},
        "gpus": {"available": False, "devices": [], "processes": []},
        "processes": [
            {
                "pid": 99,
                "argv": ["python", "kd-sensing-train", "--token", "raw-token", f"output.run_name={run_dir.name}"],
                "config_path": "https://user:pass@example.com/config.yaml",
                "run_name": run_dir.name,
                "credential": "must-not-survive",
            }
        ],
    }

    index = build_run_index(outputs=outputs, logs=None, resources=resources, now=NOW)
    payload = json.dumps(index)

    assert "raw-token" not in payload
    assert "user:pass" not in payload
    assert "must-not-survive" not in payload
    assert index["resources"]["processes"][0]["argv"][3] == "<redacted>"


def test_run_index_renderers_include_expected_fields(tmp_path: Path):
    outputs = tmp_path / "outputs"
    _write_started_run(outputs / "scene31", "render_run")

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)

    table = render_run_table(index)
    csv_text = render_run_csv(index)

    assert "state" in table
    assert "render_run" in table
    assert "run_dir,run_name,state" in csv_text


def test_run_status_sidecar_records_running_complete_and_failed(tmp_path: Path):
    cfg = _base_cfg("status_run")
    cfg.setdefault("runtime", {})["cli_config_path"] = "configs/status.yaml"
    run_dir = tmp_path / "status_run"

    running = write_running_status(run_dir, cfg, kind="training", started_at=NOW)
    complete = write_complete_status(
        run_dir,
        cfg,
        kind="training",
        primary_metric={"name": "val_adba", "value": 0.5},
        metrics_path=run_dir / "metrics.json",
        final_test_metrics_path=run_dir / "final_test_metrics.json",
        best_checkpoint=run_dir / "checkpoints" / "best.pth",
        completed_at=NOW + dt.timedelta(seconds=10),
    )

    assert running["state"] == "running"
    assert complete["state"] == "complete"
    assert complete["duration_seconds"] == 10.0
    assert complete["final_test_metrics_path"].endswith("final_test_metrics.json")
    assert json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))["primary_metric"]["value"] == 0.5

    failed_dir = tmp_path / "failed_run"
    write_running_status(failed_dir, cfg, kind="evaluation", started_at=NOW)
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        failed = write_failed_status_for_active_run(cfg, exc, kind="evaluation")

    assert failed is not None
    assert failed["state"] == "failed"
    assert failed["exception"]["type"] == "RuntimeError"


def test_run_card_writes_json_and_markdown_without_checkpoint_contents(tmp_path: Path):
    outputs = tmp_path / "outputs"
    run_dir = _write_started_run(outputs / "scene31", "card_run")
    (run_dir / "metrics.json").write_text(json.dumps({"val_adba": 0.42}), encoding="utf-8")
    (run_dir / "train_log.json").write_text(json.dumps({"epoch_logs": [{"epoch": 1}]}), encoding="utf-8")
    (run_dir / "training_outputs.npz").write_bytes(b"placeholder")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pth").write_bytes(b"secret checkpoint bytes")
    (checkpoint_dir / "best.pth.json").write_text(json.dumps({"selection_metric": "val_adba", "selected_epoch": 1}), encoding="utf-8")

    index = build_run_index(outputs=outputs, logs=None, include_resources=False, now=NOW)
    card = build_run_card(index["runs"][0], project_root=tmp_path, now=NOW)
    paths = write_run_card(card, output_dir=tmp_path / "cards")

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert payload["schema_version"] == 1
    assert payload["checkpoint"]["path"].endswith("checkpoints/best.pth")
    assert payload["checkpoint"]["selection_metric"] == "val_adba"
    assert payload["provenance"]["run_id"]
    assert "secret checkpoint bytes" not in json.dumps(payload)
    assert "secret checkpoint bytes" not in markdown
