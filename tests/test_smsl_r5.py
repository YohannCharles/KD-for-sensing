import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd_sensing.baselines.smsl import (
    F2_FEATURE_NAMES,
    directional_margin_distillation,
    legal_f2_features,
    normalized_hard_weights,
    normalized_risk_weights,
    severe_availability,
    shuffled_weights,
    validate_legal_feature_names,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_smsl_r5 import (  # noqa: E402
    _enforce_resource_contract,
    _meaningful_primary_gain,
    _resource_launch_command,
    _scheduled_f2_alpha,
    _stable_seed,
    _subset_loader,
    _success_checks,
    _weight_for_arm,
    build_f2,
)


def test_legal_f2_features_depend_only_on_missing_view_tensors() -> None:
    availability = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    logits = torch.tensor([[3.0, 1.0, -1.0], [0.0, 1.0, 2.0]])
    embedding = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    unimodal = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
            [[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]],
        ]
    )
    features = legal_f2_features(availability, logits, embedding, unimodal)
    assert features.shape == (2, len(F2_FEATURE_NAMES))
    assert features[:, :4].tolist() == [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]]
    assert features[:, 4].tolist() == pytest.approx([2.0, 1.0])
    assert features[:, 6].tolist() == pytest.approx([5.0, 2.0])
    assert torch.isfinite(features).all()


def test_f2_feature_contract_rejects_label_dependent_name() -> None:
    validate_legal_feature_names()
    with pytest.raises(ValueError, match="legal"):
        validate_legal_feature_names(F2_FEATURE_NAMES + ("missing_target_rank",))


def test_risk_and_hard_weights_are_detached_bounded_and_normalized() -> None:
    risk = torch.tensor([0.05, 0.5, 0.95], requires_grad=True)
    weights = normalized_risk_weights(risk, alpha=1.0)
    assert not weights.requires_grad
    assert weights.min() >= 0.5
    assert weights.max() <= 2.0
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)
    hard = normalized_hard_weights(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))
    assert not hard.requires_grad
    assert hard.mean().item() == pytest.approx(1.0, abs=1e-6)


def test_shuffled_control_preserves_weight_distribution() -> None:
    values = torch.tensor([0.5, 0.75, 1.25, 2.0])
    shuffled = shuffled_weights(values, generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(shuffled.sort().values, values.sort().values)
    assert not torch.equal(shuffled, values)


def test_severe_scope_is_one_or_two_available_modalities() -> None:
    availability = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    assert severe_availability(availability).tolist() == [False, False, True, True]


def test_directional_margin_uses_only_teacher_correct_samples_and_detaches_teacher() -> None:
    student = torch.tensor([[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]], requires_grad=True)
    result = directional_margin_distillation(student, teacher, torch.tensor([0, 2]))
    assert result["teacher_correct"].tolist() == [True, False]
    assert result["confuser"].tolist() == [1, 1]
    assert result["loss"].item() == pytest.approx(3.0)
    result["loss"].backward()
    assert teacher.grad is None
    assert student.grad is not None


def test_directional_margin_applies_detached_sufficiency_weight_only_to_teacher_correct_rows() -> None:
    student = torch.tensor([[1.0, 2.0, 0.0], [1.0, 0.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]], requires_grad=True)
    weights = torch.tensor([2.0, 100.0], requires_grad=True)
    result = directional_margin_distillation(student, teacher, torch.tensor([0, 2]), weights=weights)
    assert result["loss"].item() == pytest.approx(6.0)
    result["loss"].backward()
    assert teacher.grad is None
    assert weights.grad is None


def test_f2_weight_schedule_is_uniform_for_exact_first_ten_percent_of_steps() -> None:
    values = [_scheduled_f2_alpha(step, 100, alpha=1.0, warmup_fraction=0.1) for step in range(100)]
    assert values[:10] == [0.0] * 10
    assert values[10:] == [1.0] * 90


def test_fixed_mask_control_normalizes_only_the_severe_rows() -> None:
    availability = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1]], dtype=torch.bool)
    weights, _ = _weight_for_arm(
        "c1",
        torch.ones(4),
        severe_availability(availability),
        torch.full((4,), 0.5),
        availability,
        torch.linspace(0.05, 0.95, 14),
        alpha=1.0,
        config={"f2": {"weight_clamp": [0.5, 2.0]}},
    )
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)
    assert weights[2:].tolist() == [1.0, 1.0]


def test_arm_loader_order_uses_its_own_seed_instead_of_global_rng_state() -> None:
    base = DataLoader(TensorDataset(torch.arange(20)), batch_size=4, shuffle=False)
    first = _subset_loader(base, list(range(20)), workers=0, shuffle=True, seed=2026)
    first_order = torch.cat([batch[0] for batch in first])
    _ = torch.rand(100)
    second = _subset_loader(base, list(range(20)), workers=0, shuffle=True, seed=2026)
    second_order = torch.cat([batch[0] for batch in second])
    torch.testing.assert_close(first_order, second_order)


def test_formal_resource_contract_is_reproducible_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    config = {
        "runtime": {
            "dataloader_workers": 8,
            "torch_intraop_threads": 1,
            "torch_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": environment,
        }
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    snapshot = _enforce_resource_contract(config, phase="screening", workers=8)
    assert snapshot["enforced"]
    assert snapshot["torch_intraop_threads"] == 1
    command = _resource_launch_command(config, 3, ["tools/run_smsl_r5.py", "--phase", "screening"])
    assert "CUDA_VISIBLE_DEVICES=3" in command
    assert "OMP_NUM_THREADS=1" in command
    assert "--no-capture-output -n kd_mm_beam" in command
    with pytest.raises(RuntimeError, match="requires --workers 8"):
        _enforce_resource_contract(config, phase="screening", workers=0)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="thread contract mismatch"):
        _enforce_resource_contract(config, phase="screening", workers=8)


def test_existing_frozen_f2_validation_is_read_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "artifacts/f2/f2_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "fit_source_roles": ["train"],
            "feature_contract": "missing_view_only_v1",
            "state": {
                "mean": [0.0] * len(F2_FEATURE_NAMES),
                "scale": [1.0] * len(F2_FEATURE_NAMES),
                "weight": [0.0] * len(F2_FEATURE_NAMES),
                "bias": 0.0,
            },
        },
        checkpoint,
    )
    audit = tmp_path / "f2_audit.json"
    audit.write_text('{"sentinel": "frozen"}\n', encoding="utf-8")
    original = audit.read_bytes()
    result = build_f2({}, tmp_path)
    assert result["sentinel"] == "frozen"
    assert audit.read_bytes() == original


def test_phase2_success_checks_apply_every_preregistered_threshold() -> None:
    baseline = {
        "single_worst_top1": 0.10,
        "severe_worst_top1": 0.10,
        "severe_top1_macro": 0.40,
        "all14_worst_top1": 0.10,
        "single_within3_worst": 0.50,
        "severe_within3_worst": 0.50,
        "single_mae_worst": 10.0,
        "severe_mae_worst": 10.0,
        "full_top1": 0.86,
        "severe_g1_far_error_macro": 0.30,
    }
    thresholds = {
        "single_worst_gain": 0.02,
        "severe_worst_gain": 0.015,
        "severe_macro_gain": 0.01,
        "all14_worst_gain": 0.015,
        "within3_max_drop": 0.005,
        "mae_max_increase": 0.05,
        "full_top1_max_drop": 0.005,
        "far_error_requires_decrease": True,
    }
    candidate = {
        **baseline,
        "single_worst_top1": 0.12,
        "severe_worst_top1": 0.115,
        "severe_top1_macro": 0.41,
        "all14_worst_top1": 0.115,
        "single_within3_worst": 0.495,
        "severe_within3_worst": 0.495,
        "single_mae_worst": 10.05,
        "severe_mae_worst": 10.05,
        "full_top1": 0.855,
        "severe_g1_far_error_macro": 0.299,
    }
    checks = _success_checks(candidate, baseline, thresholds)
    assert all(checks.values())
    candidate["all14_worst_top1"] -= 1e-4
    assert not _success_checks(candidate, baseline, thresholds)["all14_worst_gain"]


def test_primary_attribution_requires_meaningful_gain_without_other_worst_case_regression() -> None:
    thresholds = {"single_worst_gain": 0.02, "severe_worst_gain": 0.015}
    reference = {"single_worst_top1": 0.10, "severe_worst_top1": 0.20}
    assert _meaningful_primary_gain({"single_worst_top1": 0.12, "severe_worst_top1": 0.20}, reference, thresholds)
    assert not _meaningful_primary_gain({"single_worst_top1": 0.12, "severe_worst_top1": 0.199}, reference, thresholds)
    assert _stable_seed("a3", "Severe", "top1", "worst") == _stable_seed("a3", "Severe", "top1", "worst")
