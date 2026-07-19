import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from kd_sensing.data.mmw.twc_router_joint_training import prepare_router_joint_training_panel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


WATCH = _load("launch_mmw_dynamic_router_evaluation")
SUMMARY = _load("summarize_mmw_router_joint_stress")
JOINT = _load("launch_mmw_router_joint_stress")
EVALUATOR = _load("eval_mmw_router_joint_stress")


def _training_manifest(
    tmp_path: Path,
    *,
    name: str = "primary",
    candidates=None,
    status: str = "running",
    job_status: str = "running",
    numeric_policy: bool = False,
) -> Path:
    candidates = WATCH.CANDIDATES if candidates is None else candidates
    source_config = tmp_path / "source" / "CurrentControl.yaml"
    source_checkpoint = tmp_path / "source" / "last.pth"
    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text("source: current\n", encoding="utf-8")
    source_checkpoint.write_bytes(b"source-checkpoint")
    panel = tmp_path / name / "panel.json"
    panel_payload = prepare_router_joint_training_panel(panel)
    panel_checksum = str(panel_payload["checksum"])
    jobs = []
    for candidate, variant, supervision in candidates:
        gpu = next(index for index, item in enumerate(WATCH.CANDIDATES) if item[0] == candidate)
        run_dir = tmp_path / name / "runs" / candidate / "seed1"
        config = tmp_path / name / "configs" / f"{candidate}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        screen = {
            "protocol": WATCH.TRAINING_PROTOCOL_ID,
            "candidate": candidate,
            "router_variant": variant,
            "supervision": supervision,
            "seed": 1,
            "source_checkpoint_sha256": WATCH._sha256(source_checkpoint),
            "joint_panel_checksum": panel_checksum,
            "selection_split": "frozen_inner_validation_only",
            "claim_eligible": False,
        }
        if numeric_policy:
            screen.update(
                utility_numeric_policy=WATCH.UTILITY_NUMERIC_POLICY,
                router_reliability_source_sha256=WATCH._sha256(WATCH.ROUTER_RELIABILITY_SOURCE),
            )
        config.write_text(
            yaml.safe_dump(
                {
                    "experiment": {"name": candidate, "seed": 1, "ablation_id": candidate},
                    "model": {"primary": {"router_variant": variant}},
                    "loss": {
                        "u_mask_beam_jepa": {
                            "dynamic_router": {
                                "supervision": supervision,
                                "paired_joint": {
                                    "panel_path": str(panel.resolve()),
                                    "panel_sha256": panel_checksum,
                                },
                            }
                        }
                    },
                    "training": {
                        "initialization_checkpoint": {
                            "path": str(source_checkpoint),
                            "sha256": WATCH._sha256(source_checkpoint),
                        }
                    },
                    "output": {"dir": str(run_dir.parent), "run_name": "seed1"},
                    "mmw_dynamic_router_screen": screen,
                }
            ),
            encoding="utf-8",
        )
        jobs.append(
            {
                "candidate": candidate,
                "router_variant": variant,
                "supervision": supervision,
                "seed": 1,
                "gpu": gpu,
                "config_path": str(config),
                "config_sha256": WATCH._sha256(config),
                "run_dir": str(run_dir),
                "status": job_status,
                "return_code": 0 if job_status == "done" else None,
                "claim_eligible": False,
            }
        )
    payload = {
        "protocol": WATCH.TRAINING_PROTOCOL_ID,
        "request": {
            "protocol": WATCH.TRAINING_PROTOCOL_ID,
            "panel_protocol": "mmw_router_joint_training_v1",
            "panel_checksum": panel_checksum,
            "panel_seed": 20260719,
            "source_config": str(source_config),
            "source_config_sha256": WATCH._sha256(source_config),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": WATCH._sha256(source_checkpoint),
            "seed": 1,
            "batch_size": 64,
            "epochs": 10,
            "gpus": [next(index for index, item in enumerate(WATCH.CANDIDATES) if item[0] == candidate) for candidate, _, _ in candidates],
            "candidates": [list(item) for item in candidates],
            "selection_split": "frozen_inner_validation_only",
            "claim_eligible": False,
        },
        "request_sha256": "",
        "panel_path": str(panel),
        "status": status,
        "jobs": jobs,
    }
    if numeric_policy:
        payload["request"].update(
            utility_numeric_policy=WATCH.UTILITY_NUMERIC_POLICY,
            router_reliability_source_sha256=WATCH._sha256(WATCH.ROUTER_RELIABILITY_SOURCE),
        )
    payload["request_sha256"] = WATCH._payload_sha256(payload["request"])
    path = tmp_path / name / "training_manifest_seed1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_complete_artifact(job: dict) -> None:
    run_dir = Path(job["run_dir"])
    checkpoint = run_dir / "checkpoints/last.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(job["candidate"].encode())
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "config_path": str(Path(job["config_path"]).resolve()),
                "run_dir": str(run_dir.resolve()),
                "experiment_name": job["candidate"],
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )
    (Path(str(checkpoint) + ".json")).write_text(
        json.dumps(
            {
                "publish_complete": True,
                "checkpoint_role": "last",
                "checkpoint_schema_version": 1,
                "checkpoint_sha256": WATCH._sha256(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "path": str(checkpoint.resolve()),
                "run_dir": str(run_dir.resolve()),
            }
        ),
        encoding="utf-8",
    )


def test_watcher_waits_for_complete_training_artifacts_and_preserves_gpu_mapping(tmp_path: Path) -> None:
    training_path = _training_manifest(tmp_path)
    plan = WATCH.prepare_plan(
        training_manifest=training_path,
        output_root=tmp_path / "evaluation",
        cache=tmp_path / "cache.json",
        bootstrap_iterations=100,
        child_poll_seconds=1.0,
    )
    orchestration = json.loads(plan.read_text(encoding="utf-8"))
    state, detail = WATCH._training_state(orchestration)
    assert state == "waiting"
    assert detail["manifests"] == {"primary": "running"}
    assert [(job["candidate"], job["gpu"]) for job in orchestration["jobs"]] == [
        (candidate, gpu) for gpu, (candidate, _, _) in enumerate(WATCH.CANDIDATES)
    ]

    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["status"] = "complete"
    for job in training["jobs"]:
        job["status"] = "done"
        job["return_code"] = 0
        run_dir = Path(job["run_dir"])
        _write_complete_artifact(job)
    training_path.write_text(json.dumps(training), encoding="utf-8")

    state, detail = WATCH._training_state(orchestration)
    assert state == "ready"
    assert len(detail["artifacts"]) == 8
    command = WATCH._candidate_command(orchestration["jobs"][3], orchestration["request"])
    assert command[command.index("--gpus") + 1] == "3"
    assert "--allow-gpu0-3" in command
    assert command[-2:] == ["--retry-failed", "--launch"]


def test_watcher_fails_closed_when_training_manifest_failed(tmp_path: Path) -> None:
    training_path = _training_manifest(tmp_path, status="failed", job_status="failed")
    plan = WATCH.prepare_plan(
        training_manifest=training_path,
        output_root=tmp_path / "evaluation",
        cache=tmp_path / "cache.json",
        bootstrap_iterations=100,
        child_poll_seconds=1.0,
    )
    state, detail = WATCH._training_state(json.loads(plan.read_text(encoding="utf-8")))
    assert state == "failed"
    assert detail["reason"] == "selected_training_job_failed"


def test_watcher_recognizes_manually_completed_candidate_output(tmp_path: Path) -> None:
    config = tmp_path / "candidate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mmw_dynamic_router_screen": {
                    "candidate": "PATR-Label",
                    "router_variant": "patr",
                    "supervision": "label_topology",
                    "claim_eligible": False,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "PATR-Label"
    output.mkdir()
    cache = tmp_path / "cache.json"
    cache.write_text('{"checksum":"cache-checksum"}', encoding="utf-8")
    checkpoint = tmp_path / "last.pth"
    checkpoint.write_bytes(b"checkpoint")
    config_sha = WATCH._sha256(config)
    checkpoint_sha = WATCH._sha256(checkpoint)
    candidate_provenance = WATCH._config_router_provenance(config)
    outer_request = {
        "joint_stress_protocol": WATCH.EVALUATOR_PROTOCOL,
        "evaluator_algorithm": WATCH.EVALUATOR_ALGORITHM,
        "branch_algorithm": WATCH.BRANCH_ALGORITHM,
        "fusion_branches": list(WATCH.DYNAMIC_FUSIONS),
        "cache": str(cache),
        "cache_sha256": WATCH._sha256(cache),
        "cache_checksum": "cache-checksum",
        "oracle_helper_sha256": WATCH._sha256(WATCH.ROOT / "scripts/eval_mmw_router_oracle_gap.py"),
        "corruption_runtime_sha256": WATCH._sha256(WATCH.ROOT / "src/kd_sensing/evaluation/corruptions.py"),
        "joint_cache_runtime_sha256": WATCH._sha256(WATCH.ROOT / "src/kd_sensing/data/mmw/twc_router_joint_stress.py"),
        "evaluator_sha256": WATCH._sha256(WATCH.ROOT / "scripts/eval_mmw_router_joint_stress.py"),
        "summary_sha256": WATCH._sha256(WATCH.ROOT / "scripts/summarize_mmw_router_joint_stress.py"),
        "candidate_sources": {"PATR-Label": {"config_sha256": config_sha, "checkpoint_sha256": checkpoint_sha}},
        **WATCH._joint_stress_request_identity(),
    }
    job = {
        "candidate": "PATR-Label",
        "config": str(config),
        "config_sha256": config_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "gpu": 0,
        "output_root": str(output),
        "attempts": 0,
    }
    eval_request = {
        "protocol": WATCH.EVALUATOR_PROTOCOL,
        "evaluator_protocol": WATCH.EVALUATOR_PROTOCOL,
        "evaluator_algorithm": outer_request["evaluator_algorithm"],
        "branch_algorithm": outer_request["branch_algorithm"],
        "fusion_branches": outer_request["fusion_branches"],
        "config": str(config.resolve()),
        "config_sha256": config_sha,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "cache": str(cache.resolve()),
        "cache_sha256": outer_request["cache_sha256"],
        "cache_checksum": outer_request["cache_checksum"],
        "evaluator_sha256": outer_request["evaluator_sha256"],
        "oracle_helper_sha256": outer_request["oracle_helper_sha256"],
        "corruption_runtime_sha256": outer_request["corruption_runtime_sha256"],
        "joint_cache_runtime_sha256": outer_request["joint_cache_runtime_sha256"],
        "summary_sha256": outer_request["summary_sha256"],
        "claim_eligible": False,
        "split": "frozen_inner_validation_only",
        "router_candidate_provenance": candidate_provenance,
        "gpus": [0],
        "orchestration_attempt": "",
        **WATCH._joint_stress_request_identity(),
    }
    request_sha = WATCH._payload_sha256(eval_request)
    shard_jobs = []
    for index in range(8):
        marker = output / "shards" / f"s{index}" / "complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        shard = {"shard": f"s{index}", "completion_marker": str(marker), "status": "complete", "returncode": 0}
        marker.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "protocol": WATCH.EVALUATOR_PROTOCOL,
                    "shard": f"s{index}",
                    "config_sha256": config_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "cache_sha256": outer_request["cache_sha256"],
                    "cache_checksum": outer_request["cache_checksum"],
                    "evaluator_algorithm": outer_request["evaluator_algorithm"],
                    "request_sha256": request_sha,
                }
            ),
            encoding="utf-8",
        )
        shard_jobs.append(shard)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "summary": {"status": "complete"},
                "request": eval_request,
                "request_sha256": request_sha,
                "jobs": shard_jobs,
            }
        ),
        encoding="utf-8",
    )
    (output / "joint_summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "fusion_branches": list(WATCH.DYNAMIC_FUSIONS),
                    "claim_eligible": False,
                    "evaluation_request_sha256": request_sha,
                    "checkpoint_sha256": checkpoint_sha,
                    "cache_sha256": outer_request["cache_sha256"],
                    "cache_checksum": outer_request["cache_checksum"],
                    "branch_algorithm": outer_request["branch_algorithm"],
                    "evaluator_algorithm": outer_request["evaluator_algorithm"],
                    "source_sha256": {
                        "evaluator_sha256": outer_request["evaluator_sha256"],
                        "oracle_helper_sha256": outer_request["oracle_helper_sha256"],
                        "corruption_runtime_sha256": outer_request["corruption_runtime_sha256"],
                        "joint_cache_runtime_sha256": outer_request["joint_cache_runtime_sha256"],
                        "summary_sha256": outer_request["summary_sha256"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert WATCH._candidate_complete(job, outer_request) is True
    job["attempt_id"] = "frozen-attempt"
    assert WATCH._candidate_complete(job, outer_request) is False
    job.pop("attempt_id")
    manifest_path = output / "evaluation_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["request"]["corruption_seed"] += 1
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert WATCH._candidate_complete(job, outer_request) is False


def test_candidate_reconcile_rejects_stale_output_after_failed_attempt(monkeypatch) -> None:
    monkeypatch.setattr(WATCH, "_candidate_complete", lambda job, request: True)
    job = {
        "status": "failed",
        "attempt_id": "frozen-attempt",
        "failed_attempt_id": "frozen-attempt",
        "attempts": 1,
        "returncode": 7,
    }

    WATCH._reconcile_candidate_job(job, {})

    assert job["status"] == "failed"
    assert job["returncode"] == 7


def test_candidate_reconcile_never_consumes_stale_output_while_child_alive(monkeypatch) -> None:
    monkeypatch.setattr(WATCH, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(WATCH, "_candidate_complete", lambda job, request: True)
    job = {"status": "running", "pid": 42, "attempt_id": "frozen-attempt", "attempts": 2}

    WATCH._reconcile_candidate_job(job, {})

    assert job["status"] == "running"


@pytest.mark.parametrize(("attempts", "expected"), ((1, "planned"), (2, "failed")))
def test_candidate_reconcile_downgrades_invalid_complete_state(monkeypatch, attempts: int, expected: str) -> None:
    monkeypatch.setattr(WATCH, "_candidate_complete", lambda job, request: False)
    job = {"status": "complete", "pid": None, "attempts": attempts, "returncode": 0}

    WATCH._reconcile_candidate_job(job, {})

    assert job["status"] == expected


def test_watcher_merges_primary_labels_with_same_name_power_repair(tmp_path: Path) -> None:
    power_candidates = tuple(item for item in WATCH.CANDIDATES if item[2] == "beam_power")
    primary_path = _training_manifest(tmp_path, status="failed", job_status="running")
    repair_path = _training_manifest(
        tmp_path,
        name="repair",
        candidates=power_candidates,
        status="complete",
        job_status="done",
        numeric_policy=True,
    )
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    for job in primary["jobs"]:
        if job["supervision"] == "beam_power":
            job.update(status="failed", return_code=-15)
        else:
            job["status"] = "done"
            job["return_code"] = 0
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    for path in (primary_path, repair_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for job in payload["jobs"]:
            if job["status"] != "done":
                continue
            _write_complete_artifact(job)

    plan = WATCH.prepare_plan(
        training_manifest=primary_path,
        repair_training_manifest=repair_path,
        output_root=tmp_path / "evaluation",
        cache=tmp_path / "cache.json",
        bootstrap_iterations=100,
        child_poll_seconds=1.0,
    )
    orchestration = json.loads(plan.read_text(encoding="utf-8"))

    primary_request_sha = json.loads(primary_path.read_text(encoding="utf-8"))["request_sha256"]
    repair_request_sha = json.loads(repair_path.read_text(encoding="utf-8"))["request_sha256"]
    assert orchestration["request"]["training_manifests"] == {
        "primary": {"path": str(primary_path), "request_sha256": primary_request_sha},
        "repair": {"path": str(repair_path), "request_sha256": repair_request_sha},
    }
    assert {
        job["candidate"]: job["training_source"]["role"] for job in orchestration["jobs"]
    } == {
        candidate: ("repair" if supervision == "beam_power" else "primary")
        for candidate, _, supervision in WATCH.CANDIDATES
    }
    state, detail = WATCH._training_state(orchestration)
    assert state == "ready"
    assert {item["source_role"] for item in detail["artifacts"]} == {"primary", "repair"}


def test_power_repair_numeric_policy_and_source_sha_are_fail_closed(tmp_path: Path) -> None:
    power_candidates = tuple(item for item in WATCH.CANDIDATES if item[2] == "beam_power")
    primary_path = _training_manifest(tmp_path)
    repair_path = _training_manifest(
        tmp_path,
        name="repair",
        candidates=power_candidates,
        numeric_policy=True,
    )
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    repair["request"]["utility_numeric_policy"] = "legacy_float16_underflow"
    repair["request_sha256"] = WATCH._payload_sha256(repair["request"])
    repair_path.write_text(json.dumps(repair), encoding="utf-8")

    with pytest.raises(ValueError, match="numeric policy/source"):
        WATCH.prepare_plan(
            training_manifest=primary_path,
            repair_training_manifest=repair_path,
            output_root=tmp_path / "evaluation",
            cache=tmp_path / "cache.json",
            bootstrap_iterations=100,
            child_poll_seconds=1.0,
        )


def test_training_panel_semantic_checksum_is_recomputed(tmp_path: Path) -> None:
    training_path = _training_manifest(tmp_path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    panel_path = Path(training["panel_path"])
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["conditions"][0]["state_matrix"][0][0] = 99
    panel_path.write_text(json.dumps(panel), encoding="utf-8")

    with pytest.raises(ValueError, match="training panel"):
        WATCH.prepare_plan(
            training_manifest=training_path,
            output_root=tmp_path / "evaluation",
            cache=tmp_path / "cache.json",
            bootstrap_iterations=100,
            child_poll_seconds=1.0,
        )


def test_power_repair_recipe_diff_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    power_candidates = tuple(item for item in WATCH.CANDIDATES if item[2] == "beam_power")
    primary_path = _training_manifest(tmp_path)
    repair_path = _training_manifest(
        tmp_path,
        name="repair",
        candidates=power_candidates,
        numeric_policy=True,
    )
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    config_path = Path(repair["jobs"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["lr"] = 0.123
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repair["jobs"][0]["config_sha256"] = WATCH._sha256(config_path)
    repair_path.write_text(json.dumps(repair), encoding="utf-8")

    with pytest.raises(ValueError, match="narrow numeric-fix allowlist"):
        WATCH.prepare_plan(
            training_manifest=primary_path,
            repair_training_manifest=repair_path,
            output_root=tmp_path / "evaluation",
            cache=tmp_path / "cache.json",
            bootstrap_iterations=100,
            child_poll_seconds=1.0,
        )


def test_joint_launcher_freezes_legacy_or_dynamic_branch_inventory(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.yaml"
    dynamic = tmp_path / "dynamic.yaml"
    legacy.write_text("{}\n", encoding="utf-8")
    dynamic.write_text("mmw_dynamic_router_screen:\n  claim_eligible: false\n", encoding="utf-8")

    assert JOINT._config_fusion_branches(legacy) == JOINT.LEGACY_FUSIONS
    assert JOINT._config_fusion_branches(dynamic) == JOINT.DYNAMIC_FUSIONS


def test_joint_launcher_and_evaluator_share_algorithm_identity() -> None:
    assert JOINT.EVALUATOR_ALGORITHM == EVALUATOR.EVALUATOR_ALGORITHM


def test_dynamic_controls_separate_reference_and_post_health_experts() -> None:
    reference = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    candidate = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])
    prior = torch.tensor([[0.75, 0.25]])
    available = torch.ones(1, 2, dtype=torch.bool)

    controls = EVALUATOR.dynamic_control_logits(
        candidate_unimodal=candidate,
        reference_unimodal=reference,
        static_prior=prior,
        available=available,
    )

    assert torch.allclose(controls["uniform_logits"], torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(controls["train_fit_static_prior_logits"], torch.tensor([[1.5, 0.5]]))
    assert torch.allclose(controls["post_health_uniform_logits"], torch.tensor([[2.0, 0.5]]))
    assert torch.allclose(controls["post_health_static_prior_logits"], torch.tensor([[3.0, 0.25]]))

    identical = EVALUATOR.dynamic_control_logits(
        candidate_unimodal=reference,
        reference_unimodal=reference,
        static_prior=prior,
        available=available,
    )
    assert torch.equal(identical["uniform_logits"], identical["post_health_uniform_logits"])
    assert torch.equal(
        identical["train_fit_static_prior_logits"], identical["post_health_static_prior_logits"]
    )


def test_dynamic_main_uniform_and_oracle_use_reference_experts() -> None:
    reference = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    candidate = torch.tensor([[[0.0, 5.0], [5.0, 0.0]]])
    prior = torch.tensor([[0.5, 0.5]])
    available = torch.ones(1, 2, dtype=torch.bool)
    powers = torch.tensor([[1.0, 0.1]])

    controls = EVALUATOR.dynamic_control_branches(
        candidate_unimodal=candidate,
        reference_unimodal=reference,
        static_prior=prior,
        available=available,
        beam_powers=powers,
    )

    assert torch.equal(controls["uniform_logits"], torch.tensor([[2.5, 2.5]]))
    assert torch.equal(controls["oracle_logits"], reference[:, 0])
    assert torch.equal(controls["post_health_uniform_logits"], candidate.mean(dim=1))
    assert torch.equal(controls["candidate_oracle_logits"], candidate[:, 1])


def test_domain_rows_scope_router_regret_to_learned_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        EVALUATOR,
        "metrics_for",
        lambda *args, **kwargs: {"adba": 0.5},
    )
    trace = {
        "target": torch.tensor([0]).numpy(),
        "beam_powers": torch.ones(1, 2).numpy(),
        "learned_logits": torch.zeros(1, 2).numpy(),
        "uniform_logits": torch.zeros(1, 2).numpy(),
        "oracle_logits": torch.zeros(1, 2).numpy(),
        "router_weights": torch.tensor([[0.6, 0.4]]).numpy(),
        "uniform_weights": torch.tensor([[0.5, 0.5]]).numpy(),
        "router_soft_oracle_regret": torch.tensor([0.12]).numpy(),
        "router_selection_oracle_regret": torch.tensor([0.08]).numpy(),
    }
    rows = EVALUATOR.domain_metric_rows(
        type("Model", (), {"modalities": ("image", "radar")})(),
        {},
        {
            "pattern": "clean",
            "condition_index": 0,
            "requested_stress_rate": 0.0,
            "drop_rate": 0.0,
            "corrupt_rate": 0.0,
            "mask_set_index": -1,
            "state_digest": "clean",
        },
        {"id": "d0", "condition": "clear", "scene": "Town03"},
        trace,
        cache_checksum="cache",
    )
    by_fusion = {row["fusion"]: row for row in rows}
    assert by_fusion["learned"]["router_soft_oracle_regret"] == pytest.approx(0.12)
    assert by_fusion["learned"]["router_regret_scope"] == "learned_router_branch_only"
    assert by_fusion["uniform"]["router_soft_oracle_regret"] is None
    assert by_fusion["uniform"]["router_regret_scope"] == "not_applicable_control_branch"


def test_dynamic_response_uses_static_prior_cell_baseline() -> None:
    trace = {
        "router_residual_logits": torch.zeros(2, 2).numpy(),
        "router_weights": torch.tensor([[0.8, 0.2], [0.8, 0.2]]).numpy(),
        "static_prior_weights": torch.tensor([[0.8, 0.2], [0.8, 0.2]]).numpy(),
        "router_effective_cell_weights": torch.tensor(
            [
                [[0.2, 0.1], [0.6, 0.1]],
                [[0.5, 0.1], [0.3, 0.1]],
            ]
        ).numpy(),
        "state_matrix": torch.tensor(
            [
                [[2, 0], [0, 0]],
                [[2, 0], [0, 0]],
            ],
            dtype=torch.int8,
        ).numpy(),
    }

    result = EVALUATOR.dynamic_response_diagnostics(trace)

    assert result["corrupted_cell_static_prior_mass"] == pytest.approx(0.4)
    assert result["corrupted_cell_weight_mass"] == pytest.approx(0.35)
    assert result["corrupted_cell_weight_vs_static_ratio"] == pytest.approx(0.875)
    assert result["corrupted_cell_downweight_vs_static_rate"] == pytest.approx(0.5)
    assert result["corrupted_cell_weight_response_ratio"] == pytest.approx(1.4)


def test_dynamic_gate_uses_train_fit_prior_and_clean_noninferiority() -> None:
    rate_rows = []
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
        rate_rows.append(
            {
                "requested_stress_rate": rate,
                "learned_minus_uniform_adba": -0.1,
                "learned_minus_uniform_normalized_gain": -0.1,
                "learned_minus_train_fit_static_prior_adba": -0.001 if rate == 0.0 else 0.02,
                "learned_minus_train_fit_static_prior_normalized_gain": 0.02,
                "learned_minus_frozen_current_router_adba": -0.002 if rate == 0.0 else 0.01,
                "learned_minus_frozen_current_router_normalized_gain": 0.01,
                "corrupted_cell_downweight_vs_static_rate": 0.7,
                "corrupted_cell_weight_vs_static_ratio": 0.9,
            }
        )
    intervals = [
        {
            "scope": "Joint40_60_80Combined",
            "control": control,
            "metric": metric,
            "mean_delta": 0.02,
            "ci_low": 0.01,
        }
        for control in ("train_fit_static_prior", "frozen_current_router", "uniform")
        for metric in SUMMARY.GATE_METRICS
    ]

    gate = SUMMARY._gate_decision(rate_rows, intervals, fusions=SUMMARY.DYNAMIC_FUSIONS)

    assert gate["passed"] is True
    assert gate["primary_control"] == "train_fit_static_prior"
    assert gate["claim_eligible"] is False
    assert gate["decision"] == "advance_to_pure_drop"
    assert gate["pure_drop_protection"].startswith("pending")


@pytest.mark.parametrize(
    "failure",
    ("per_rate", "combined_effect", "current_ci", "clean", "downweight_rate", "weight_ratio"),
)
def test_dynamic_gate_fails_each_frozen_threshold(failure: str) -> None:
    rates = [
        {
            "requested_stress_rate": rate,
            "learned_minus_train_fit_static_prior_adba": 0.01,
            "learned_minus_train_fit_static_prior_normalized_gain": 0.01,
            "learned_minus_frozen_current_router_adba": 0.01,
            "learned_minus_frozen_current_router_normalized_gain": 0.01,
            "corrupted_cell_downweight_vs_static_rate": 0.7,
            "corrupted_cell_weight_vs_static_ratio": 0.9,
        }
        for rate in (0.0, 0.2, 0.4, 0.6, 0.8)
    ]
    intervals = [
        {
            "scope": "Joint40_60_80Combined",
            "control": control,
            "metric": metric,
            "mean_delta": 0.01,
            "ci_low": 0.005,
        }
        for control in ("train_fit_static_prior", "frozen_current_router", "uniform")
        for metric in SUMMARY.GATE_METRICS
    ]
    rates = deepcopy(rates)
    intervals = deepcopy(intervals)
    if failure == "per_rate":
        rates[3]["learned_minus_train_fit_static_prior_adba"] = 0.0
    elif failure == "combined_effect":
        next(
            row
            for row in intervals
            if row["control"] == "train_fit_static_prior" and row["metric"] == "adba"
        )["mean_delta"] = 0.0019
    elif failure == "current_ci":
        next(
            row
            for row in intervals
            if row["control"] == "frozen_current_router" and row["metric"] == "normalized_gain"
        )["ci_low"] = 0.0
    elif failure == "clean":
        rates[0]["learned_minus_frozen_current_router_adba"] = -0.0021
    elif failure == "downweight_rate":
        for row in rates:
            row["corrupted_cell_downweight_vs_static_rate"] = 0.59
    else:
        for row in rates:
            row["corrupted_cell_weight_vs_static_ratio"] = 1.0

    gate = SUMMARY._gate_decision(rates, intervals, fusions=SUMMARY.DYNAMIC_FUSIONS)

    assert gate["passed"] is False
    assert gate["decision"] == "candidate_rejected_by_inner_gate"


def test_dynamic_bootstrap_reports_static_current_and_uniform_controls() -> None:
    rows = []
    offsets = {
        "learned": 0.04,
        "train_fit_static_prior": 0.01,
        "frozen_current_router": 0.02,
        "uniform": 0.0,
    }
    for rate in (0.0, 0.2, 0.4, 0.6, 0.8):
        for domain in range(15):
            for fusion, offset in offsets.items():
                rows.append(
                    {
                        "requested_stress_rate": rate,
                        "domain_id": f"d{domain}",
                        "fusion": fusion,
                        "adba": offset + domain * 1e-4,
                        "normalized_gain": offset + domain * 1e-4,
                    }
                )

    first = SUMMARY._bootstrap_rows(
        rows,
        controls=SUMMARY._bootstrap_controls(SUMMARY.DYNAMIC_FUSIONS),
        iterations=200,
        seed=7,
        confidence=0.95,
    )
    second = SUMMARY._bootstrap_rows(
        rows,
        controls=SUMMARY._bootstrap_controls(SUMMARY.DYNAMIC_FUSIONS),
        iterations=200,
        seed=7,
        confidence=0.95,
    )

    assert first == second
    combined = {
        (row["control"], row["metric"]): row
        for row in first
        if row["scope"] == "Joint40_60_80Combined"
    }
    assert set(combined) == {
        (control, metric)
        for control in ("train_fit_static_prior", "frozen_current_router", "uniform")
        for metric in SUMMARY.GATE_METRICS
    }
    assert combined[("train_fit_static_prior", "adba")]["mean_delta"] == pytest.approx(0.03)


def test_summary_fusion_inventory_must_match_manifest(tmp_path: Path) -> None:
    condition_dir = tmp_path / "clean"
    condition_dir.mkdir()
    rows = "fusion\n" + "\n".join(SUMMARY.DYNAMIC_FUSIONS) + "\n"
    (condition_dir / "domain_metrics.csv").write_text(rows, encoding="utf-8")
    manifest = {"request": {"fusion_branches": list(SUMMARY.DYNAMIC_FUSIONS)}}

    assert SUMMARY._resolve_fusions(tmp_path, {"clean": {}}, manifest) == SUMMARY.DYNAMIC_FUSIONS
    manifest["request"]["fusion_branches"] = list(SUMMARY.FUSIONS)
    with pytest.raises(ValueError, match="immutable manifest"):
        SUMMARY._resolve_fusions(tmp_path, {"clean": {}}, manifest)


def test_dynamic_summary_core_aggregates_all_conditions_and_seven_branches() -> None:
    conditions = {
        "clean": {"requested_stress_rate": 0.0, "mask_set_index": -1, "state_digest": "clean"}
    }
    for rate in SUMMARY.JOINT_RATES:
        for mask in range(SUMMARY.MASKS_PER_RATE):
            pattern = f"joint{int(rate * 100)}_m{mask:02d}"
            conditions[pattern] = {
                "requested_stress_rate": rate,
                "mask_set_index": mask,
                "state_digest": pattern,
            }
    offsets = {
        "uniform": 0.0,
        "learned": 0.04,
        "oracle": 0.08,
        "train_fit_static_prior": 0.01,
        "frozen_current_router": 0.02,
        "post_health_uniform": 0.015,
        "post_health_static_prior": 0.025,
    }
    raw = []
    for pattern, condition in conditions.items():
        for domain in range(15):
            for fusion in SUMMARY.DYNAMIC_FUSIONS:
                row = {
                    "pattern": pattern,
                    "requested_stress_rate": condition["requested_stress_rate"],
                    "mask_set_index": condition["mask_set_index"],
                    "state_digest": condition["state_digest"],
                    "domain_id": f"d{domain}",
                    "fusion": fusion,
                    "sample_count": 10,
                    **{metric: offsets[fusion] for metric in SUMMARY.METRICS},
                    **{field: 0.1 for field in SUMMARY.REGRET_FIELDS},
                    **{field: 0.25 for field in SUMMARY.ROUTER_WEIGHT_FIELDS},
                    "router_residual_abs_mean": 0.2,
                    "corrupted_cell_weight_mass": 0.3,
                    "corrupted_cell_available_share": 0.4,
                    "corrupted_cell_weight_response_ratio": 0.75,
                    "corrupted_cell_static_prior_mass": 0.4,
                    "corrupted_cell_weight_vs_static_ratio": 0.75,
                    "corrupted_cell_downweight_vs_static_rate": 0.75,
                }
                raw.append(row)

    condition_rows = SUMMARY._condition_summary(raw, conditions, fusions=SUMMARY.DYNAMIC_FUSIONS)
    domain_rows = SUMMARY._domain_rate_summary(raw, conditions, fusions=SUMMARY.DYNAMIC_FUSIONS)
    rate_rows = SUMMARY._rate_summary(domain_rows, condition_rows, fusions=SUMMARY.DYNAMIC_FUSIONS)
    bootstrap = SUMMARY._bootstrap_rows(
        domain_rows,
        controls=SUMMARY._bootstrap_controls(SUMMARY.DYNAMIC_FUSIONS),
        iterations=100,
        seed=7,
        confidence=0.95,
    )
    gate = SUMMARY._gate_decision(rate_rows, bootstrap, fusions=SUMMARY.DYNAMIC_FUSIONS)

    assert len(condition_rows) == 81 * len(SUMMARY.DYNAMIC_FUSIONS)
    assert len(domain_rows) == 5 * 15 * len(SUMMARY.DYNAMIC_FUSIONS)
    assert len(rate_rows) == 5
    assert rate_rows[-1]["post_health_static_prior_adba"] == pytest.approx(0.025)
    assert rate_rows[-1]["corrupted_cell_weight_response_ratio"] == pytest.approx(0.75)
    assert gate["passed"] is True
    markdown = SUMMARY._markdown(
        {
            "fusion_branches": list(SUMMARY.DYNAMIC_FUSIONS),
            "protocol": "test",
            "cache_sha256": "x",
            "cache_checksum": "y",
            "checkpoint_sha256": "z",
            "corruption_seed": 1,
            "corruption_parameters": {},
            "bootstrap": {"iterations": 100, "seed": 7},
            "gate_thresholds": {},
        },
        rate_rows,
        bootstrap,
        gate,
    )
    assert "Post-health static" in markdown
    assert "动态 Router 对照" in markdown
