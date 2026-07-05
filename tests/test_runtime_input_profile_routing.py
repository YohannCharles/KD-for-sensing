import pytest
import torch

from kd_sensing.engine import runtime


SINGLE_MODALITY_CASES = (
    ("radar", "prepare_radar_inputs", "radar_batch"),
    ("gps", "prepare_gps_inputs", "gps_batch"),
    ("lidar", "prepare_lidar_inputs", "lidar_batch"),
    ("mmwave", "prepare_mmwave_inputs", "mmwave_batch"),
    ("csi", "prepare_csi_inputs", "csi_batch"),
)


@pytest.mark.parametrize(("task", "helper_name", "output_key"), SINGLE_MODALITY_CASES)
def test_prepare_task_inputs_routes_same_modality_input_profile(monkeypatch, task, helper_name, output_key) -> None:
    calls: list[dict] = []
    helper_output = torch.tensor([1.0])

    def fake_prepare_inputs(batch, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return helper_output

    monkeypatch.setattr(runtime, helper_name, fake_prepare_inputs)
    input_profiles = {modality: f"{modality}_profile" for modality, _, _ in SINGLE_MODALITY_CASES}

    result = runtime.prepare_task_inputs(
        {},
        task,
        model_cfg={"input_profiles": input_profiles, "gps_input_seq_len": 2},
        seq_length=3,
        num_pred=1,
        device=torch.device("cpu"),
    )

    assert result[output_key] is helper_output
    assert calls[0]["profile"] == f"{task}_profile", f"{task} should use input_profiles.{task}"


@pytest.mark.parametrize(("task", "helper_name", "output_key"), SINGLE_MODALITY_CASES)
def test_prepare_task_inputs_missing_profile_uses_helper_default(monkeypatch, task, helper_name, output_key) -> None:
    calls: list[dict] = []
    helper_output = torch.tensor([1.0])

    def fake_prepare_inputs(batch, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return helper_output

    monkeypatch.setattr(runtime, helper_name, fake_prepare_inputs)
    other_profiles = {
        modality: f"{modality}_profile"
        for modality, _, _ in SINGLE_MODALITY_CASES
        if modality != task
    }

    result = runtime.prepare_task_inputs(
        {},
        task,
        model_cfg={"input_profiles": other_profiles, "gps_input_seq_len": 2},
        seq_length=3,
        num_pred=1,
        device=torch.device("cpu"),
    )

    assert result[output_key] is helper_output
    assert "profile" in calls[0], f"{task} should pass an explicit profile argument"
    assert calls[0]["profile"] is None, f"{task} should not fall back to another modality profile"
