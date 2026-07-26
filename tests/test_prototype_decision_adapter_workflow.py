import json
import numpy as np
import pytest
import sys
import torch
import torch.nn.functional as F

from kd_sensing.baselines.prototype_decision_adapter import (
    ADAPTER_LOSS_PROFILES,
    NON_FULL_MASKS,
    adapter_training_loss,
    generate_mask_schedule,
    numpy_metrics,
    schedule_masks,
    state_digest,
    split_permutation,
    stratified_mask_folds,
    load_u0_artifact_config,
    _sequential,
)
from torch.utils.data import DataLoader, TensorDataset
from kd_sensing.evaluation.metrics import beam_classification_circular_summary, calculate_dba_score
from kd_sensing.engine.trainer_runtime_helpers import training_loss_early_stop_state
from tools.run_full_pool_capacity import (
    ADBA_SURROGATE_JOBS,
    CONVERGENCE_EPOCHS,
    STAGE2_GPUS,
    TRAINING_LOSS_EARLY_STOPPING,
    MASK_BIAS_ABLATION_EPOCHS,
    MASK_BIAS_ALL_SEEN_JOBS,
    MASK_BIAS_UNSEEN_JOBS,
    CIRCULAR_TRANSPORT_EPOCHS,
    CIRCULAR_TRANSPORT_JOBS,
    apply_reference_u0_profile,
    choose_epochs,
    physical_gpu_uuid,
    prototype_health,
    run_jobs,
)


def test_cross_entropy_loss_profile_preserves_original_adapter_objective() -> None:
    torch.manual_seed(5)
    logits = torch.randn(4, 64)
    labels = torch.tensor([0, 7, 31, 63])
    delta_logits = torch.randn(4, 64)

    expected = F.cross_entropy(logits.float(), labels)
    expected = expected + 1e-4 * delta_logits.float().pow(2).sum(dim=1).mean()
    actual = adapter_training_loss(
        logits,
        labels,
        delta_logits,
        loss_profile="cross_entropy",
    )

    assert actual == pytest.approx(expected)


def test_adba_surrogate_prefers_near_and_circular_predictions() -> None:
    labels = torch.tensor([0])
    delta_logits = torch.zeros(1, 64)
    near = torch.full((1, 64), -6.0)
    wrapped = near.clone()
    far = near.clone()
    near[0, 1] = 6.0
    wrapped[0, 63] = 6.0
    far[0, 32] = 6.0

    near_loss = adapter_training_loss(near, labels, delta_logits, loss_profile="adba_surrogate")
    wrapped_loss = adapter_training_loss(wrapped, labels, delta_logits, loss_profile="adba_surrogate")
    far_loss = adapter_training_loss(far, labels, delta_logits, loss_profile="adba_surrogate")

    assert near_loss == pytest.approx(wrapped_loss, abs=1e-7)
    assert near_loss < far_loss


def test_adba_surrogate_profile_and_gpu_mapping_are_preregistered() -> None:
    assert ADAPTER_LOSS_PROFILES["adba_surrogate"] == {
        "hard_ce_weight": 0.5,
        "soft_ce_weight": 0.5,
        "soft_label_sigma": 2.0,
        "circular": True,
        "delta_logit_weight": 1e-4,
    }
    assert ADBA_SURROGATE_JOBS == {
        "b1": ("a1", 0),
        "b4": ("a4", 4),
        "b6": ("a6", 6),
        "b7": ("a7", 7),
    }


def test_mask_schedule_is_deterministic_balanced_and_excludes_full() -> None:
    sample_ids = [f"train:{index}" for index in range(101)]
    left = generate_mask_schedule(sample_ids, epochs=20, seed=1)
    right = generate_mask_schedule(list(reversed(sample_ids)), epochs=20, seed=1)

    assert left == right
    assert tuple(left["modality_order"]) == ("image", "radar", "gps", "lidar")
    assert all(tuple(mask) in NON_FULL_MASKS and tuple(mask) != (1, 1, 1, 1) for mask in left["masks"])
    for indices in left["mask_indices_by_epoch"]:
        counts = np.bincount(indices, minlength=14)
        assert counts.max() - counts.min() <= 1
    masks = schedule_masks(left, 3, sample_ids[:4], torch.device("cpu"))
    assert masks.shape == (4, 4)


def test_preregistered_unseen_fold_has_zero_training_exposure() -> None:
    folds = stratified_mask_folds(seed=1, fold_count=4)
    flattened = [mask for fold in folds for mask in fold]
    held_out = set(folds[0])
    allowed = tuple(mask for mask in NON_FULL_MASKS if mask not in held_out)
    schedule = generate_mask_schedule([f"sample:{index}" for index in range(103)], epochs=8, seed=1, allowed_masks=allowed)

    assert len(flattened) == len(set(flattened)) == 14
    assert set(flattened) == set(NON_FULL_MASKS)
    assert all(sum(mask) in (1, 2, 3) for mask in flattened)
    assert all(len([mask for mask in fold if sum(mask) == 1]) == 1 for fold in folds)
    assert all(len([mask for mask in fold if sum(mask) == 3]) == 1 for fold in folds)
    assert not held_out & {tuple(mask) for mask in schedule["masks"]}
    for indices in schedule["mask_indices_by_epoch"]:
        counts = np.bincount(indices, minlength=len(allowed))
        assert counts.max() - counts.min() <= 1


def test_mask_bias_ablation_job_mapping_is_preregistered() -> None:
    assert MASK_BIAS_ABLATION_EPOCHS == 8
    assert MASK_BIAS_ALL_SEEN_JOBS == {
        "global_bias": ("global_bias", 0),
        "mask_lookup": ("mask_lookup", 4),
    }
    assert MASK_BIAS_UNSEEN_JOBS == {
        "mask_mlp": ("a1", 6),
        "factorized_bias": ("factorized_bias", 7),
    }


def test_circular_transport_job_mapping_is_preregistered() -> None:
    assert CIRCULAR_TRANSPORT_EPOCHS == 8
    assert CIRCULAR_TRANSPORT_JOBS == {
        "circular_transport": ("circular_transport", 0),
        "factorized_all_seen": ("factorized_all_seen", 4),
    }


def test_a7_permutation_is_deterministic_and_split_local() -> None:
    train = [f"train:{index}" for index in range(40)]
    validation = [f"validation:{index}" for index in range(10)]
    mapping = split_permutation(train, seed=1, split="train")

    assert mapping == split_permutation(train, seed=1, split="train")
    assert set(mapping) == set(train)
    assert set(mapping.values()) == set(train)
    assert not set(mapping.values()) & set(validation)
    assert mapping != split_permutation(train, seed=1, split="validation")


def test_numpy_metrics_match_project_metric_implementation() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(37, 64)).astype(np.float32)
    target = rng.integers(0, 64, size=37)
    independent = numpy_metrics(logits, target)
    project = beam_classification_circular_summary(torch.from_numpy(logits), torch.from_numpy(target))

    assert independent["top1"] == pytest.approx(project["top1"], abs=1e-7)
    assert independent["top3"] == pytest.approx(project["top3"], abs=1e-7)
    assert independent["top5"] == pytest.approx(project["top5"], abs=1e-7)
    assert independent["within3"] == pytest.approx(project["within_3"], abs=1e-7)
    assert independent["mae"] == pytest.approx(project["mean_error"], abs=1e-7)
    adba = calculate_dba_score(
        torch.from_numpy(logits).unsqueeze(1), torch.from_numpy(target).unsqueeze(1)
    )[0]
    assert independent["adba"] == pytest.approx(float(adba), abs=1e-7)


def test_legacy_capacity_reference_mode_must_be_disabled(tmp_path, monkeypatch) -> None:
    from kd_sensing.baselines import prototype_decision_adapter as workflow

    monkeypatch.setattr(workflow, "normalize_loaded_config", lambda config: None)
    monkeypatch.setattr(workflow, "validate_loaded_config", lambda config: None)
    monkeypatch.setattr(workflow, "load_clean_inner_protocol", lambda path: {
        "protocol_id": "test", "protocol_fingerprint": "fingerprint"
    })
    monkeypatch.setattr(workflow, "DEFAULT_CONFIG", {
        "model": {"primary": {}}, "data_protocol": {"path": "protocol.yaml"}
    })
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  primary:\n    capacity_reference_mode: true\n"
        "data_protocol:\n  path: protocol.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only accepted when explicitly false"):
        load_u0_artifact_config(config)


def test_state_digest_supports_scalar_integer_buffers() -> None:
    module = torch.nn.BatchNorm1d(3)
    first = state_digest(module)
    module.num_batches_tracked.add_(1)

    assert state_digest(module) != first


def test_prototype_cache_loader_uses_accelerated_inference_settings() -> None:
    source = DataLoader(TensorDataset(torch.arange(300)), batch_size=64, num_workers=0)
    cache_loader = _sequential(source)

    assert cache_loader.batch_size == 128
    assert cache_loader.num_workers == 8
    assert cache_loader.drop_last is False


def test_full_pool_reference_profile_is_scratch_bfloat16_training() -> None:
    cfg = {
        "model": {"primary": {"router_use_pattern_features": True}},
        "temporal_missing": {"seed": 0},
        "training": {"resume": True, "validation": {}},
        "scheduler": {"type": "none"},
    }

    apply_reference_u0_profile(cfg, epochs=7)

    assert cfg["model"]["primary"]["router_use_pattern_features"] is False
    assert cfg["temporal_missing"]["seed"] == 1
    assert cfg["training"]["epochs"] == cfg["training"]["max_epochs"] == 7
    assert cfg["training"]["resume"] is False
    assert cfg["training"]["amp"] == {"enabled": True, "dtype": "bfloat16", "grad_scaler": False}
    assert cfg["training"]["early_stopping"] == {
        **TRAINING_LOSS_EARLY_STOPPING,
        "monitor": "train_task_loss",
    }
    assert cfg["training"]["weight_decay"] == pytest.approx(3e-4)
    assert cfg["scheduler"]["type"] == "cosine_warm_restarts"


def test_full_pool_epochs_prioritize_convergence_over_wall_budget() -> None:
    selection = choose_epochs({}, elapsed=123)

    assert selection["u0_epochs"] == CONVERGENCE_EPOCHS == 20
    assert selection["adapter_epochs"] == CONVERGENCE_EPOCHS
    assert selection["wall_budget_seconds"] is None
    assert selection["basis"]["validation_used_for_epoch_selection"] is False


def test_prototype_health_fails_closed_for_collapsed_bank() -> None:
    healthy = prototype_health({"prototype_bank.prototypes": torch.eye(64)})
    collapsed = prototype_health({"prototype_bank.prototypes": torch.ones(64, 8)})

    assert healthy["status"] == "passed"
    assert collapsed["status"] == "failed"
    assert collapsed["off_diagonal_cosine"]["mean"] >= collapsed["collapse_threshold"]


def test_training_loss_early_stopping_uses_pre_registered_relative_plateau() -> None:
    state = training_loss_early_stop_state(
        [1.0, 0.8, 0.7, 0.6, 0.5, 0.499, 0.4985, 0.498],
        TRAINING_LOSS_EARLY_STOPPING,
    )

    assert state["should_stop"] is True
    assert state["actual_epochs"] == 8
    assert state["epochs_without_improvement"] == 3
    assert state["stop_reason"] == "training_loss_plateau"


def test_full_pool_stage2_uses_only_three_user_supplied_gpus() -> None:
    assert STAGE2_GPUS == {
        "a0": 0,
        "a1": 0,
        "a3": 0,
        "a2": 4,
        "a4": 4,
        "a6": 6,
        "a5": 6,
        "a7": 7,
    }


def test_running_adapter_process_discovers_only_conda_wrapper(tmp_path) -> None:
    from tools import run_full_pool_capacity as workflow

    wrapper = tmp_path / "123"
    wrapper.mkdir()
    wrapper.joinpath("cmdline").write_bytes(
        b"/workspace/miniconda3/bin/python\0/workspace/miniconda3/condabin/conda\0run\0"
        b"python\0tools/run_full_pool_capacity.py\0--adapter\0a7\0"
    )
    worker = tmp_path / "124"
    worker.mkdir()
    worker.joinpath("cmdline").write_bytes(
        b"python\0tools/run_full_pool_capacity.py\0--adapter\0a7\0"
    )

    assert workflow._running_adapter_process("a7", proc_root=tmp_path) == {
        "name": "a7",
        "status": "running",
        "pid": 123,
    }


def test_full_pool_gpu_binding_uses_physical_uuid_and_rejects_unapproved_gpu(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "GPU-physical-four\n"

    monkeypatch.setattr("tools.run_full_pool_capacity.subprocess.run", lambda *args, **kwargs: Result())

    assert physical_gpu_uuid(0) == "GPU-physical-four"
    assert physical_gpu_uuid(4) == "GPU-physical-four"
    assert physical_gpu_uuid(7) == "GPU-physical-four"
    with pytest.raises(ValueError, match="refusing GPU 1"):
        physical_gpu_uuid(1)


def test_stage2_resume_adopts_only_manifest_running_jobs(tmp_path, monkeypatch) -> None:
    from tools import run_full_pool_capacity as workflow

    selection = {
        "u0_epochs": 20,
        "adapter_epochs": 20,
        "early_stopping": {"min_epochs": 8},
    }
    (tmp_path / "timing_estimate.json").write_text(
        json.dumps({"epoch_selection": selection, "u0_epoch_wall_seconds": 1.0}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "u0_seed1/checkpoints/last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "u0_seed1/early_stopping.json").write_text(
        json.dumps({"actual_epochs": 20}), encoding="utf-8"
    )
    (tmp_path / "u0_seed1/final_config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (tmp_path / "stage2").mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime.joinpath("orchestration_manifest.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": "a1", "status": "running", "pid": 123, "gpu": 0},
                    {"name": "a0", "status": "completed", "pid": 122, "gpu": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_stage2(*args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(workflow, "run_stage2", fake_run_stage2)

    assert workflow.continue_stage2(tmp_path, resume=True) == 0
    assert captured["resume"] is True
    assert set(captured["external_jobs"]) == {"a1"}


def test_run_jobs_serializes_only_jobs_sharing_a_gpu(tmp_path, monkeypatch) -> None:
    from tools import run_full_pool_capacity as workflow

    monkeypatch.setattr(workflow, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(workflow, "physical_gpu_uuid", lambda gpu: f"GPU-{gpu}")
    monkeypatch.setattr(workflow, "_gpu_status", lambda gpu: {"gpu": gpu, "memory_used_mib": 0})
    jobs = [
        {
            "name": "first",
            "gpu": 1,
            "command": [sys.executable, "-c", "import time; time.sleep(0.05)"],
            "log_path": tmp_path / "first.log",
        },
        {
            "name": "parallel",
            "gpu": 2,
            "command": [sys.executable, "-c", "import time; time.sleep(0.05)"],
            "log_path": tmp_path / "parallel.log",
        },
        {
            "name": "queued",
            "gpu": 1,
            "command": [sys.executable, "-c", "import time; time.sleep(0.01)"],
            "log_path": tmp_path / "queued.log",
        },
    ]

    result = run_jobs("test", jobs, tmp_path)

    assert all(job["return_code"] == 0 for job in result)
    assert result[2]["start_time"] >= result[0]["end_time"]
    assert result[1]["start_time"] < result[0]["end_time"]
