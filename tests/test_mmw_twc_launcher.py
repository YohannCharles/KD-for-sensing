import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_launcher(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts" / "launch_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location("mmw_twc_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict:
    return {
        "protocol_id": "mmw_twc_outer_v1",
        "protocol_kind": "post_selection_confirmation_not_historical_blind_test",
        "manifest_sha256": "a" * 64,
        "fixed_mask_cache": {"sha256": "b" * 64, "cache_checksum": "c" * 64},
    }


def _topology_manifest(launcher, tmp_path: Path) -> tuple[Path, dict]:
    descriptor = {"topology_id": "ula_dft_phase_cycle_v1", "codebook": "test_ula_dft"}
    payload = {
        "descriptor": descriptor,
        "descriptor_sha256": launcher._sha256_payload(descriptor),
    }
    path = tmp_path / "topology_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _domains() -> list[dict[str, str]]:
    return [
        {
            "id": f"weather{index // 5}/scene{index}",
            "condition": f"weather{index // 5}",
            "scene": f"scene{index}",
            "data_root": "dataset/MMW/sunny",
            "train_csv_name": f"/tmp/twc_train_{index}.csv",
            "val_csv_name": f"/tmp/twc_validation_{index}.csv",
            "test_csv_name": f"/tmp/twc_outer_{index}.csv",
        }
        for index in range(15)
    ]


def _build_config(launcher, tmp_path: Path, variant: str) -> tuple[dict, dict]:
    all_weather = launcher._all_weather_launcher()
    topology_path, topology_payload = _topology_manifest(launcher, tmp_path)
    topology = {
        "path": str(topology_path),
        "descriptor": topology_payload["descriptor"],
        "descriptor_sha256": topology_payload["descriptor_sha256"],
    }
    config = launcher.build_confirmation_config(
        all_weather,
        variant,
        tmp_path / "outputs",
        seed=3,
        batch_size=64,
        epochs=40,
        domains=_domains(),
        protocol=_protocol(),
        confirmation_splits={"manifest_sha256": "d" * 64},
        topology=topology,
    )
    return config, topology


def _prepare_plan_dependencies(launcher, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    topology_path, _ = _topology_manifest(launcher, tmp_path)
    confirmation_splits = {"manifest_sha256": "d" * 64}
    monkeypatch.setattr(launcher, "load_protocol", lambda _: deepcopy(_protocol()))
    monkeypatch.setattr(
        launcher,
        "build_confirmation_train_domains",
        lambda _protocol_manifest, _output_root: (deepcopy(_domains()), deepcopy(confirmation_splits)),
    )
    monkeypatch.setattr(launcher, "_all_weather_launcher", lambda: object())
    monkeypatch.setattr(
        launcher,
        "build_confirmation_config",
        lambda _all_weather, variant, _output_root, **kwargs: {"variant": variant, "seed": kwargs["seed"]},
    )
    return protocol_path, topology_path


def _prepare_real_config_plan_dependencies(launcher, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    protocol_path = tmp_path / "protocol_manifest.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    topology_path, _ = _topology_manifest(launcher, tmp_path)
    monkeypatch.setattr(launcher, "load_protocol", lambda _: deepcopy(_protocol()))
    monkeypatch.setattr(
        launcher,
        "build_confirmation_train_domains",
        lambda _protocol_manifest, _output_root: (deepcopy(_domains()), {"manifest_sha256": "d" * 64}),
    )
    return protocol_path, topology_path


def test_launcher_defaults_to_gpu4_to_gpu7_and_requires_explicit_gpu0_to_gpu3_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher = _load_launcher(monkeypatch)

    assert launcher._validate_gpu_ids((4, 7)) == (4, 7)
    for invalid in ((), (0,), (4, 4), (8,)):
        with pytest.raises(ValueError, match="subset of 4,5,6,7"):
            launcher._validate_gpu_ids(invalid)
    assert launcher._validate_gpu_ids((0, 4, 7), allow_gpu0_3=True) == (0, 4, 7)
    for invalid in ((), (0, 0), (8,)):
        with pytest.raises(ValueError, match="with --allow-gpu0-3"):
            launcher._validate_gpu_ids(invalid, allow_gpu0_3=True)
    # The queue performs the same check before reading a plan or touching a GPU.
    with pytest.raises(ValueError, match="subset of 4,5,6,7"):
        launcher.run_queue(tmp_path / "does-not-exist.json", gpus=(0,), min_free_mib=1, poll_seconds=1.0, max_jobs=0)

    protocol_path, topology_path = _prepare_plan_dependencies(launcher, monkeypatch, tmp_path)
    calls: dict[str, object] = {}
    actual_prepare_plan = launcher.prepare_plan

    def fake_prepare_plan(output_root, **kwargs):
        calls["output_root"] = output_root
        calls.update(kwargs)
        return tmp_path / "planned.json"

    monkeypatch.setattr(launcher, "prepare_plan", fake_prepare_plan)
    monkeypatch.setattr(launcher, "_resolve_topology_path", lambda _: topology_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_mmw_twc_evidence.py",
            "--output-root",
            str(tmp_path / "output"),
            "--protocol-manifest",
            str(protocol_path),
            "--gpus",
            "4,7",
            "--batch-size",
            "64",
        ],
    )
    assert launcher.main() == 0
    assert calls["batch_size"] == 64
    assert calls["evaluation_enabled"] is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_mmw_twc_evidence.py",
            "--output-root",
            str(tmp_path / "output"),
            "--protocol-manifest",
            str(protocol_path),
            "--gpus",
            "0,4,7",
            "--allow-gpu0-3",
            "--batch-size",
            "64",
        ],
    )
    assert launcher.main() == 0

    monkeypatch.setattr(sys, "argv", ["launch_mmw_twc_evidence.py", "--gpus", "0"])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2

    monkeypatch.setattr(sys, "argv", ["launch_mmw_twc_evidence.py", "--batch-size", "63"])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2

    monkeypatch.setattr(sys, "argv", ["launch_mmw_twc_evidence.py", "--batch-size", "32"])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2

    monkeypatch.setattr(launcher, "prepare_plan", actual_prepare_plan)
    with pytest.raises(ValueError, match="multiple of 16"):
        launcher.prepare_plan(
            tmp_path / "invalid_batch",
            protocol_path=protocol_path,
            topology_path=topology_path,
            variants=("T2",),
            seeds=(1,),
            batch_size=63,
            epochs=40,
        )
    with pytest.raises(ValueError, match="batch_size=64"):
        launcher.prepare_plan(
            tmp_path / "wrong_main_batch",
            protocol_path=protocol_path,
            topology_path=topology_path,
            variants=("T2",),
            seeds=(1,),
            batch_size=32,
            epochs=40,
        )


def test_plan_and_generated_config_tampering_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher = _load_launcher(monkeypatch)
    protocol_path, topology_path = _prepare_plan_dependencies(launcher, monkeypatch, tmp_path)
    output_root = tmp_path / "main"
    plan_path = launcher.prepare_plan(
        output_root,
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=("T2",),
        seeds=(1,),
        batch_size=64,
        epochs=40,
    )
    payload = launcher._read_json(plan_path)
    launcher._validate_plan(payload)

    tampered_plan = deepcopy(payload)
    tampered_plan["jobs"][0]["seed"] = 99
    with pytest.raises(ValueError, match="immutable identity checksum"):
        launcher._validate_plan(tampered_plan)

    config_path = Path(payload["jobs"][0]["config_path"])
    config_path.write_text(config_path.read_text(encoding="utf-8") + "tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated config checksum mismatch"):
        launcher._validate_plan(payload)


def test_plan_preflight_enforces_the_variant_specific_allowed_config_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    protocol_path, topology_path = _prepare_real_config_plan_dependencies(launcher, monkeypatch, tmp_path)
    plan_path = launcher.prepare_plan(
        tmp_path / "ablation",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=("T2-NoRouterOracle",),
        seeds=(1,),
        batch_size=64,
        epochs=40,
    )
    payload = launcher._read_json(plan_path)
    job = payload["jobs"][0]
    assert job["matched_control"] == "T2"
    assert job["allowed_config_diff"] == list(launcher.VARIANT_ALLOWED_CONFIG_DIFFS["T2-NoRouterOracle"])
    assert "generated_control_configs/T2_seed1.yaml" in job["matched_control_config_path"]
    launcher._validate_plan(payload)

    # Simulate a deliberately re-hashed plan whose ablation also changed the
    # learning rate.  Hashes alone would accept this; the registered diff
    # contract must still reject it before the queue can launch it.
    config_path = Path(job["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["lr"] = 1.0e-3
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    job["config_sha256"] = launcher._sha256_file(config_path)
    payload["plan_sha256"] = launcher._plan_sha256(payload)
    with pytest.raises(ValueError, match="outside its allowed_config_diff"):
        launcher._validate_plan(payload)


def test_registered_main_and_ablation_variants_all_pass_their_comparison_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    protocol_path, topology_path = _prepare_real_config_plan_dependencies(launcher, monkeypatch, tmp_path)
    plan_path = launcher.prepare_plan(
        tmp_path / "all_variants",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=tuple(launcher.VARIANT_PROTOCOL),
        seeds=(1,),
        batch_size=64,
        epochs=40,
    )
    payload = launcher._read_json(plan_path)
    assert payload["schema_version"] == launcher.PLAN_SCHEMA_VERSION == 3
    assert len(payload["jobs"]) == len(launcher.VARIANT_PROTOCOL)
    assert all("allowed_config_diff" in job for job in payload["jobs"])
    launcher._validate_plan(payload)

    assert launcher._resolve_variants("main", None) == (
        "T2",
        "S1",
        "masktrain_cls",
        "amber_full",
        "rmbp_mm",
        "amr_net_4m",
    )
    main_plan = launcher.prepare_plan(
        tmp_path / "four_method_main",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=launcher.MAIN_VARIANTS,
        seeds=launcher.SEEDS,
        batch_size=64,
        epochs=40,
    )
    main_payload = launcher._read_json(main_plan)
    assert len(main_payload["jobs"]) == 30


def test_smoke_skips_outer_evaluation_while_main_jobs_stay_planned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher = _load_launcher(monkeypatch)
    protocol_path, topology_path = _prepare_plan_dependencies(launcher, monkeypatch, tmp_path)

    smoke_plan = launcher.prepare_plan(
        tmp_path / "smoke",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=("T2",),
        seeds=(1,),
        batch_size=64,
        epochs=1,
        smoke=True,
    )
    smoke_job = launcher._read_json(smoke_plan)["jobs"][0]
    assert smoke_job["status"] == "planned"
    assert smoke_job["evaluation_status"] == "skipped"

    main_plan = launcher.prepare_plan(
        tmp_path / "main",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=("T2",),
        seeds=(1,),
        batch_size=64,
        epochs=40,
        smoke=False,
    )
    main_job = launcher._read_json(main_plan)["jobs"][0]
    assert main_job["status"] == "planned"
    assert main_job["evaluation_status"] == "planned"

    train_only_plan = launcher.prepare_plan(
        tmp_path / "train_only",
        protocol_path=protocol_path,
        topology_path=topology_path,
        variants=("T2",),
        seeds=(1,),
        batch_size=64,
        epochs=40,
        evaluation_enabled=False,
    )
    train_only_job = launcher._read_json(train_only_plan)["jobs"][0]
    assert train_only_job["status"] == "planned"
    assert train_only_job["evaluation_status"] == "skipped"


def test_orphaned_training_resumes_in_place_and_failed_evaluations_require_explicit_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    run_dir = tmp_path / "T2" / "seed5"
    checkpoint = run_dir / "checkpoints" / "last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    job = {
        "method": "T2",
        "seed": 5,
        "status": "running",
        "pid": 123,
        "gpu": 0,
        "run_dir": str(run_dir),
        "config_path": str(tmp_path / "T2_seed5.yaml"),
        "log_path": str(tmp_path / "T2_seed5.log"),
        "evaluation_status": "failed",
        "evaluation_return_code": 1,
        "evaluation_end_time": "before",
    }
    monkeypatch.setattr(launcher, "_completed_run", lambda _job: False)
    monkeypatch.setattr(launcher, "_completed_evaluation", lambda _job: False)
    monkeypatch.setattr(launcher, "_pid_is_running", lambda _pid: False)

    launcher._recover_orphaned_jobs([job])

    assert job["status"] == "planned"
    assert job["pid"] is None
    assert job["resume_from_orphaned_checkpoint"] is True
    assert job["evaluation_status"] == "failed"
    assert launcher._retry_failed_evaluations([job]) is True
    assert job["evaluation_status"] == "planned"
    assert job["evaluation_retry_history"] == [{"at": job["evaluation_retry_requested_at"], "previous_return_code": 1, "previous_end_time": "before"}]

    calls: dict[str, object] = {}

    class _Process:
        pid = 456

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    process, handle = launcher._start_training_job(job, gpu=0)
    handle.close()

    assert process.pid == 456
    assert calls["command"][-1] == "--auto-resume"
    assert calls["kwargs"]["start_new_session"] is True


def test_orphan_without_last_checkpoint_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher = _load_launcher(monkeypatch)
    job = {"status": "running", "pid": 123, "run_dir": str(tmp_path / "T2" / "seed5"), "evaluation_status": "planned"}
    monkeypatch.setattr(launcher, "_completed_run", lambda _job: False)
    monkeypatch.setattr(launcher, "_pid_is_running", lambda _pid: False)

    launcher._recover_orphaned_jobs([job])

    assert job["status"] == "failed"
    assert "no resumable" in job["recovery_failure"]


def test_queue_drains_live_manifest_jobs_before_returning_a_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launcher = _load_launcher(monkeypatch)
    jobs = [{"status": "done", "evaluation_status": "failed"}]
    checks = iter((True, False))
    writes: list[object] = []
    monkeypatch.setattr(launcher, "_read_json", lambda _path: {"jobs": jobs})
    monkeypatch.setattr(launcher, "_validate_plan", lambda _manifest: None)
    monkeypatch.setattr(launcher, "_recover_orphaned_jobs", lambda _jobs: None)
    monkeypatch.setattr(launcher, "_refresh_completed_jobs", lambda _jobs: None)
    monkeypatch.setattr(launcher, "_has_live_manifest_jobs", lambda _jobs: next(checks))
    monkeypatch.setattr(launcher, "_write_json", lambda *_args: writes.append(object()))
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)

    assert launcher.run_queue(tmp_path / "manifest.json", gpus=(4,), min_free_mib=1, poll_seconds=0.1, max_jobs=0) == 1
    assert len(writes) >= 2


def test_live_manifest_jobs_reserve_their_gpu_across_scheduler_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_launcher(monkeypatch)
    jobs = [
        {"status": "running", "pid": 101, "gpu": 4, "evaluation_status": "planned"},
        {"status": "done", "evaluation_status": "running", "evaluation_pid": 102, "evaluation_gpu": 5},
        {"status": "running", "pid": 103, "gpu": None, "evaluation_status": "planned"},
    ]
    monkeypatch.setattr(launcher, "_pid_is_running", lambda pid: pid in {101, 102})

    assert launcher._occupied_manifest_gpus(jobs) == {4, 5}


def test_t2_and_s1_share_the_frozen_h4_router_recipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    t2, topology = _build_config(launcher, tmp_path / "t2", "T2")
    s1, _ = _build_config(launcher, tmp_path / "s1", "S1")

    for config in (t2, s1):
        protocol = config["mmw_all_weather_protocol"]
        assert protocol["training_profile"]["id"] == "umask_h4_v1"
        assert protocol["router_architecture_profile"]["id"] == "umask_router_nopattern_v1"
        assert config["training"]["optimizer"] == {"type": "adamw"}
        assert config["training"]["weight_decay"] == pytest.approx(3.0e-4)
        assert config["scheduler"] == {"type": "cosine_warm_restarts", "T_0": 40, "T_mult": 1, "eta_min": 1.0e-6}
        assert config["model"]["primary"]["router_use_pattern_features"] is False
        assert config["training"]["epochs"] == config["training"]["max_epochs"] == 40
        assert config["mmw_twc_evidence"]["topology_id"] == "ula_dft_phase_cycle_v1"
        assert config["mmw_twc_evidence"]["topology_descriptor_sha256"] == topology["descriptor_sha256"]
        assert config["temporal_missing"]["mode"] == "balanced_pattern_schedule"
        assert config["temporal_missing"]["schedule_id"] == "mmw_fair_pattern_v1"
        assert config["mmw_twc_evidence"]["training_mask_seed_algorithm"] == launcher.TRAINING_MASK_SEED_ALGORITHM

    assert "pattern_weighted_ce" not in t2["loss"]["u_mask_beam_jepa"]
    assert "pattern_weighted_ce" not in s1["loss"]["u_mask_beam_jepa"]


def test_whole_only_uses_a_distinct_balanced_whole_modality_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    t2, _ = _build_config(launcher, tmp_path / "t2", "T2")
    whole_only, _ = _build_config(launcher, tmp_path / "whole", "T2-WholeOnly")

    temporal = whole_only["temporal_missing"]
    assert temporal["mode"] == "balanced_pattern_schedule"
    assert temporal["schedule_id"] == "mmw_fair_whole_modality_v1"
    assert temporal["panel_size"] == 480
    assert temporal["condition_counts"] == {
        "clean": 120,
        "drop1": 120,
        "drop2": 120,
        "drop3": 120,
        "token20": 0,
        "token40": 0,
        "token60": 0,
        "token80": 0,
        "token90": 0,
    }
    assert whole_only["mmw_twc_evidence"]["training_mask_seed_algorithm"] == (
        launcher.WHOLE_ONLY_TRAINING_MASK_SEED_ALGORITHM
    )
    assert launcher.VARIANT_ALLOWED_CONFIG_DIFFS["T2-WholeOnly"] == (
        "temporal_missing.condition_counts",
        "temporal_missing.panel_size",
        "temporal_missing.schedule_id",
    )
    assert t2["temporal_missing"]["schedule_id"] == "mmw_fair_pattern_v1"
    assert t2["mmw_twc_evidence"]["training_mask_seed_algorithm"] == launcher.TRAINING_MASK_SEED_ALGORITHM


def test_bpa_topology_variants_carry_distinct_physical_linear_and_permuted_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher(monkeypatch)
    physical, topology = _build_config(launcher, tmp_path / "physical", "T2")
    linear, _ = _build_config(launcher, tmp_path / "linear", "T2-TopologyLinear")
    permuted, _ = _build_config(launcher, tmp_path / "permuted", "T2-TopologyPermuted")

    physical_loss = physical["loss"]["u_mask_beam_jepa"]
    linear_loss = linear["loss"]["u_mask_beam_jepa"]
    permuted_loss = permuted["loss"]["u_mask_beam_jepa"]
    assert physical_loss["prototype_topology"] == {
        "id": "ula_dft_phase_cycle_v1",
        "descriptor_sha256": topology["descriptor_sha256"],
        "audit_path": str((tmp_path / "physical" / "topology_manifest.json")),
    }
    assert linear_loss["prototype_topology"] == {"id": "linear_index_v1"}
    assert linear_loss["prototype_target_circular"] is False
    assert permuted_loss["prototype_topology"]["id"] == "permuted_index_v1"
    assert sorted(permuted_loss["prototype_topology"]["permutation"]) == list(range(64))
    assert permuted_loss["prototype_topology"]["permutation"] == launcher._fixed_permutation()

    evidence = [config["mmw_twc_evidence"] for config in (physical, linear, permuted)]
    assert [item["topology_id"] for item in evidence] == [
        "ula_dft_phase_cycle_v1",
        "linear_index_v1",
        "permuted_index_v1",
    ]
    assert len({item["topology_mapping_sha256"] for item in evidence}) == 3
    assert evidence[0]["topology_descriptor_sha256"] == topology["descriptor_sha256"]
    assert evidence[1]["topology_descriptor_sha256"] == "not_applicable"
    assert evidence[2]["topology_descriptor_sha256"] == "not_applicable"
