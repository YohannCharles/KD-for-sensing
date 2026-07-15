import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run_mmw_t2_bpa_cma_ablation_after_training.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_task_output(module, raw_root: Path, method: str, seed: int, count: int = 15) -> None:
    target = raw_root / f"seed{seed}" / method
    domains = target / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    names = [f"weather/domain_{index}" for index in range(count)]
    for name in names:
        (domains / f"{module._safe_name(name)}.npz").touch()
    (target / "worker_0_of_1.json").write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                "domain_shard_index": 0,
                "domain_shard_count": 1,
                "completed_domains": names,
            }
        ),
        encoding="utf-8",
    )


def test_training_state_requires_complete_status_and_last_checkpoint(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "T2-NoBPA" / "seed1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text('{"state":"complete"}\n', encoding="utf-8")

    missing = module._training_state(tmp_path, "T2-NoBPA", 1)
    assert missing["state"] == "complete"
    assert missing["ready"] is False

    checkpoint = run_dir / "checkpoints" / "last.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assert module._training_state(tmp_path, "T2-NoBPA", 1)["ready"] is True


def test_task_output_completion_requires_exact_domains_and_matching_worker(tmp_path):
    module = _load_script()
    _write_task_output(module, tmp_path, "T2-Linear", 2)

    assert module._task_output_state(tmp_path, "T2-Linear", 2)["complete"] is True
    worker = tmp_path / "seed2/T2-Linear/worker_0_of_1.json"
    payload = json.loads(worker.read_text(encoding="utf-8"))
    payload["seed"] = 3
    worker.write_text(json.dumps(payload), encoding="utf-8")
    state = module._task_output_state(tmp_path, "T2-Linear", 2)
    assert state["complete"] is False
    assert state["state"] == "incomplete"


def test_failed_training_stops_before_extract_or_summary(tmp_path, monkeypatch):
    module = _load_script()
    training_root = tmp_path / "training"
    for method in module.VARIANTS:
        for seed in module.SEEDS:
            run_dir = training_root / method / f"seed{seed}"
            run_dir.mkdir(parents=True)
            state = "failed" if (method, seed) == ("T2-CLS", 2) else "running"
            (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    monkeypatch.setattr(module, "_run_extract_jobs", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(module, "_run_summary", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    status_path = training_root / "task_output_orchestrator_status.json"

    code = module.run_orchestrator(
        training_root,
        tmp_path / "raw",
        tmp_path / "masks",
        gpus=(0, 1),
        poll_seconds=0.01,
        gpu_grace_seconds=0.0,
        status_path=status_path,
    )

    assert code == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "blocked_training_failed"
    assert "T2-CLS/seed2" in status["failures"]


def test_extract_command_is_single_full_domain_task_in_required_conda_environment(tmp_path):
    module = _load_script()
    command = module._extract_command(tmp_path / "training", tmp_path / "raw", tmp_path / "masks", "T2-CLS-CMA", 3)

    assert command[:6] == ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python"]
    assert command[6:9] == ["scripts/analyze_mmw_fused_feature_geometry.py", "extract", "--root"]
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "raw/seed3")
    assert command[command.index("--method") + 1] == "T2-CLS-CMA"
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--domain-shard-count") + 1] == "1"


def test_gpu_release_grace_is_configurable_and_defaults_to_30_seconds():
    module = _load_script()

    assert module.build_parser().parse_args([]).gpu_grace_seconds == 30.0
    assert module.build_parser().parse_args(["--gpu-grace-seconds", "5"]).gpu_grace_seconds == 5.0


def test_ready_matrix_waits_for_gpu_release_before_extract(tmp_path, monkeypatch):
    module = _load_script()
    training_root = tmp_path / "training"
    raw_root = tmp_path / "raw"
    generated = training_root / "generated_configs"
    generated.mkdir(parents=True)
    for method in module.VARIANTS:
        for seed in module.SEEDS:
            run_dir = training_root / method / f"seed{seed}"
            checkpoint = run_dir / "checkpoints" / "last.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (run_dir / "run_status.json").write_text('{"state":"complete"}\n', encoding="utf-8")
            (generated / f"{method}_seed{seed}.yaml").touch()
    for seed in module.SEEDS:
        _write_task_output(module, raw_root, module.REFERENCE_METHOD, seed)
    sleeps = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    monkeypatch.setattr(module, "_run_extract_jobs", lambda *args, **kwargs: {"error": "test stop"})

    code = module.run_orchestrator(
        training_root,
        raw_root,
        tmp_path / "masks",
        gpus=(0, 1),
        poll_seconds=1.0,
        gpu_grace_seconds=7.0,
        status_path=training_root / "task_output_orchestrator_status.json",
    )

    assert code == 3
    assert sleeps == [7.0]


def test_extract_queue_never_reuses_a_gpu_while_its_job_is_running(tmp_path, monkeypatch):
    module = _load_script()
    active = set()
    launched = []

    class ImmediateProcess:
        def __init__(self, _command, **kwargs):
            self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            assert self.gpu not in active
            active.add(self.gpu)
            launched.append(self.gpu)
            self.returncode = None

        def poll(self):
            if self.returncode is None:
                active.remove(self.gpu)
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", ImmediateProcess)
    monkeypatch.setattr(module, "_task_output_state", lambda *_args: {"complete": True, "state": "complete"})
    jobs = (("T2-NoBPA", 1), ("T2-Linear", 1), ("T2-CLS", 1))

    failure = module._run_extract_jobs(
        jobs,
        tmp_path / "training",
        tmp_path / "raw",
        tmp_path / "masks",
        gpus=(4, 5),
        log_dir=tmp_path / "logs",
        status_path=tmp_path / "status.json",
        completed=[],
    )

    assert failure is None
    assert launched == ["4", "5", "4"]
    assert not active
