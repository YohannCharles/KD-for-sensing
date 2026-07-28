import json

import numpy as np
import pytest

from kd_sensing.data.datasets.mmw import MMWDataset
from kd_sensing.data.mmw.pilot_alignment import resolve_input_channel_refs, resolve_last_input_channel_ref
from kd_sensing.data.samples import SequenceSamples


def _row(history, future):
    return {
        "history_frame_ids_json": json.dumps(history),
        "future_frame_ids_json": json.dumps(future),
    }


def _channels(tmp_path, frames):
    relative = []
    for frame in frames:
        path = tmp_path / "Channel_Data" / f"{frame}_paths.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, a=np.ones((1, 1, 2, 1, 4, 1, 1), dtype=np.complex64), tau=np.zeros((1, 1, 1, 1)))
        relative.append(str(path.relative_to(tmp_path)))
    return relative


def test_last_input_channel_is_selected_without_future_leakage(tmp_path):
    frames = [f"{value:06d}" for value in range(100, 106)]
    channels = _channels(tmp_path, frames[:5])
    result = resolve_last_input_channel_ref(
        _row(frames[:5], frames[5:]), channels, data_root=tmp_path, seq_len=5, num_pred=1
    )
    assert result["pilot_frame_id"] == "000104"
    assert result["target_frame_id"] == "000105"
    assert result["channel_ref"].endswith("000104_paths.npz")


def test_all_history_channels_are_aligned_without_future_leakage(tmp_path):
    frames = [f"{value:06d}" for value in range(100, 106)]
    channels = _channels(tmp_path, frames[:5])
    result = resolve_input_channel_refs(
        _row(frames[:5], frames[5:]), channels, data_root=tmp_path, seq_len=5, num_pred=1
    )
    assert result["history_frame_ids"] == frames[:5]
    assert [path.rsplit("/", 1)[-1] for path in result["channel_history_refs"]] == [
        f"{frame}_paths.npz" for frame in frames[:5]
    ]
    assert result["target_frame_id"] == "000105"


def test_target_channel_reference_is_rejected_before_loading(tmp_path):
    history = [f"{value:06d}" for value in range(100, 105)]
    channels = _channels(tmp_path, history[:-1] + ["000105"])
    with pytest.raises(ValueError, match="does not match history frame"):
        resolve_last_input_channel_ref(
            _row(history, ["000105"]), channels, data_root=tmp_path, seq_len=5, num_pred=1
        )


def test_non_consecutive_history_is_rejected_before_loading(tmp_path):
    history = ["000100", "000101", "000103", "000104", "000105"]
    channels = _channels(tmp_path, history)
    with pytest.raises(ValueError, match="consecutive"):
        resolve_input_channel_refs(
            _row(history, ["000106"]), channels, data_root=tmp_path, seq_len=5, num_pred=1
        )


def test_recovery_current_beam_comes_from_current_power_not_future_alias(tmp_path):
    channel = _channels(tmp_path, ["000100"])[0]
    power = np.zeros(64, dtype=np.float32)
    power[7] = 1.0
    np.savetxt(tmp_path / "beam100.txt", power)
    dataset = MMWDataset.__new__(MMWDataset)
    dataset.enabled_modalities = ()
    dataset.use_gps = False
    dataset.use_lidar = False
    dataset.include_router_utility_targets = False
    dataset.include_router_corruption_metadata = False
    dataset.include_channel_ref = True
    dataset.include_channel_history_refs = True
    dataset.data_root = tmp_path
    dataset.seq_len = 1
    dataset.num_pred = 1
    dataset.scene_slug = "scene"
    dataset.split = "train"
    dataset.condition = "sunny"
    dataset._beam_power_cache = {}
    dataset.samples = SequenceSamples(
        rgb_paths=[],
        radar_paths=[],
        channel_paths=[[channel]],
        rows=[
            {
                **_row(["000100"], ["000101"]),
                "beam1": "beam100.txt",
                "beam_label": 9,
                "future_beam_label1": 9,
            }
        ],
    )
    sample = dataset[0]
    assert sample["current_beam"].item() == 7
    assert sample["prepared_beam_label"].item() == 9
