from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.deepverse.codebook import beam_entropy, compute_beam_gain, make_ula_dft_codebook
from kd_sensing.data.deepverse.label_constants import (
    BLOCKAGE_IGNORE_INDEX,
    LINK_STATE_LOS,
    LINK_STATE_NLOS,
    LINK_STATE_UNKNOWN,
)
from kd_sensing.data.deepverse.label_scene import (
    MobilityTrace,
    _dataset_value,
    _device_id_candidates,
    _extract_paths,
    _field,
    _get_sample,
    _json_dumps,
    _json_scalar,
    _make_group_key,
    _make_sample_id,
    _metadata_value,
)
from kd_sensing.data.deepverse.label_targets import (
    RADAR_FEATURE_SIZE,
    extract_radar_feature,
    los_to_blockage,
    los_to_link_state,
    _infer_num_ant,
    _link_los_from_path_statuses,
    _parse_ue_file_range,
    _stack_label_arrays,
    _stack_or_empty,
)
from kd_sensing.data.deepverse.label_writers import write_label_cache

@dataclass
class DeepVerseLabelBuilder:
    dataset: Any
    scenario: str = "DT31"
    scenario_root: str | Path | None = None
    scenes: Sequence[int] | None = None
    seq_len: int = 8
    pred_horizon: int = 3
    num_beams: int = 64
    beam_topk: int = 5
    position_noise_std: float = 1.0
    seed: int = 42
    ue_ids: Sequence[int] | None = None
    bs_ids: Sequence[int] = (0,)
    camera_ids: Sequence[int] = (1,)
    lidar_ids: Sequence[int] = (1,)
    enable_camera: bool = True
    enable_lidar: bool = True
    enable_radar: bool = True
    blockage_min_class_count: int = 1
    blockage_min_class_ratio: float = 0.0
    skip_counts: Counter[str] = field(default_factory=Counter, init=False)
    los_status_source_counts: Counter[str] = field(default_factory=Counter, init=False)
    _raw_los_cache: dict[tuple[int, int, int], int | None] = field(default_factory=dict, init=False)
    _bs_ue_file_cache: dict[tuple[int, int], tuple[Path, int] | None] = field(default_factory=dict, init=False)

    def build(self) -> dict[str, Any]:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if self.pred_horizon <= 0:
            raise ValueError(f"pred_horizon must be positive, got {self.pred_horizon}.")
        if self.num_beams <= 1:
            raise ValueError(f"num_beams must be > 1, got {self.num_beams}.")

        self.skip_counts.clear()
        self.los_status_source_counts.clear()
        self._raw_los_cache.clear()
        self._bs_ue_file_cache.clear()

        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, Any]] = []
        labels_acc: dict[str, list[Any]] = {
            "sample_id": [],
            "beam_label": [],
            "beam_labels_future": [],
            "blockage_label": [],
            "blockage_labels_future": [],
            "blockage_valid_mask": [],
            "trajectory_future": [],
            "los_status_future": [],
            "link_state_future": [],
            "beam_gain_future": [],
            "valid_mask": [],
        }
        weak_wireless_history: list[np.ndarray] = []
        radar_feature_history: list[np.ndarray] = []
        noisy_position_history: list[np.ndarray] = []
        clean_position_history: list[np.ndarray] = []

        for ue_id in self._resolve_ue_ids():
            try:
                trace = self._mobility_for_ue(ue_id)
            except Exception:
                self.skip_counts["skipped_no_mobility"] += 1
                continue
            times = trace.times
            locations = trace.locations
            if len(times) < self.seq_len + self.pred_horizon:
                self.skip_counts["skipped_short_window"] += 1
                continue

            for bs_id in self.bs_ids:
                for anchor_local_idx in range(self.seq_len - 1, len(times) - self.pred_horizon):
                    hist_local = list(range(anchor_local_idx - self.seq_len + 1, anchor_local_idx + 1))
                    future_local = list(range(anchor_local_idx + 1, anchor_local_idx + 1 + self.pred_horizon))
                    hist_times = [times[idx] for idx in hist_local]
                    future_times = [times[idx] for idx in future_local]

                    try:
                        future_summaries = [
                            self._comm_summary(time_value, local_idx, bs_id=bs_id, ue_id=ue_id)
                            for time_value, local_idx in zip(future_times, future_local)
                        ]
                        history_summaries = [
                            self._comm_summary(time_value, local_idx, bs_id=bs_id, ue_id=ue_id)
                            for time_value, local_idx in zip(hist_times, hist_local)
                        ]
                    except Exception:
                        self.skip_counts["skipped_no_comm"] += 1
                        continue

                    beam_labels = np.asarray([summary["beam_label"] for summary in future_summaries], dtype=np.int64)
                    los_future = np.asarray([summary["los_status"] for summary in future_summaries], dtype=np.int16)
                    link_state = np.asarray([los_to_link_state(status) for status in los_future], dtype=np.int16)
                    blockage = np.asarray([los_to_blockage(status) for status in los_future], dtype=np.int64)
                    blockage_valid_mask = blockage != BLOCKAGE_IGNORE_INDEX
                    trajectory = np.asarray(locations[future_local, :2], dtype=np.float32)
                    if not (
                        np.all(np.isfinite(beam_labels))
                        and np.all(np.isfinite(trajectory))
                    ):
                        self.skip_counts["skipped_nan_label"] += 1
                        continue

                    camera_paths: list[list[str]] = []
                    if self.enable_camera:
                        camera_paths = [self._paths_for_time("camera", time_value, local_idx) for time_value, local_idx in zip(hist_times, hist_local)]
                        if any(not paths for paths in camera_paths):
                            self.skip_counts["skipped_no_camera"] += 1
                            continue

                    lidar_paths: list[list[str]] = []
                    if self.enable_lidar:
                        lidar_paths = [self._paths_for_time("lidar", time_value, local_idx) for time_value, local_idx in zip(hist_times, hist_local)]
                        if any(not paths for paths in lidar_paths):
                            self.skip_counts["skipped_no_lidar"] += 1
                            continue

                    radar_hist = np.empty((self.seq_len, 0), dtype=np.float32)
                    if self.enable_radar:
                        try:
                            radar_hist = np.stack(
                                [
                                    self._radar_feature_for_time(time_value, local_idx, bs_id=bs_id, ue_id=ue_id)
                                    for time_value, local_idx in zip(hist_times, hist_local)
                                ]
                            ).astype(np.float32)
                        except Exception:
                            self.skip_counts["skipped_no_radar"] += 1
                            continue

                    clean_hist = np.asarray(locations[hist_local, :2], dtype=np.float32)
                    noisy_hist = (
                        clean_hist
                        + rng.normal(0.0, self.position_noise_std, size=clean_hist.shape).astype(np.float32)
                    )
                    weak_hist = np.stack([summary["weak_wireless"] for summary in history_summaries]).astype(np.float32)
                    gain_future = np.stack([summary["beam_gain"] for summary in future_summaries]).astype(np.float32)
                    scene_id = _metadata_value(
                        trace.info,
                        anchor_local_idx,
                        ("scene_id", "scene", "scene_idx", "scenario_id"),
                        default=_dataset_value(self.dataset, ("scene_id", "scene", "scenario_id"), self.scenario),
                    )
                    sequence_id = _metadata_value(
                        trace.info,
                        anchor_local_idx,
                        ("sequence_id", "seq_id", "sequence", "pass_id", "track_id"),
                        default=_dataset_value(self.dataset, ("sequence_id", "seq_id", "pass_id", "track_id"), ""),
                    )
                    segment_id = _metadata_value(
                        trace.info,
                        anchor_local_idx,
                        ("segment_id", "segment", "segment_idx"),
                        default=_dataset_value(self.dataset, ("segment_id", "segment", "segment_idx"), ""),
                    )
                    object_id = _metadata_value(
                        trace.info,
                        anchor_local_idx,
                        ("object_id", "ue_id", "user_id"),
                        default=ue_id,
                    )
                    split_group_key = _make_group_key(scene_id, sequence_id, segment_id, object_id)
                    raw_frame_group_key = split_group_key

                    row = (
                        {
                            "sample_id": _make_sample_id(
                                self.scenario,
                                scene_id=scene_id,
                                sequence_id=sequence_id,
                                segment_id=segment_id,
                                object_id=object_id,
                                ue_id=ue_id,
                                bs_id=bs_id,
                                t_anchor=times[anchor_local_idx],
                            ),
                            "scenario": self.scenario,
                            "scene_id": _json_scalar(scene_id),
                            "sequence_id": _json_scalar(sequence_id),
                            "segment_id": _json_scalar(segment_id),
                            "object_id": _json_scalar(object_id),
                            "split_group_key": split_group_key,
                            "raw_frame_group_key": raw_frame_group_key,
                            "ue_id": int(ue_id),
                            "bs_id": int(bs_id),
                            "t_anchor": _json_scalar(times[anchor_local_idx]),
                            "history_indices": _json_dumps([_json_scalar(value) for value in hist_times]),
                            "future_indices": _json_dumps([_json_scalar(value) for value in future_times]),
                            "camera_paths": _json_dumps(camera_paths),
                            "lidar_paths": _json_dumps(lidar_paths),
                            "radar_feature_history": _json_dumps(radar_hist.tolist()),
                            "past_beam_indices": _json_dumps([int(summary["beam_label"]) for summary in history_summaries]),
                            "past_topk_beam_powers": _json_dumps(
                                [summary["topk_beam_powers"].tolist() for summary in history_summaries]
                            ),
                            "past_max_power": _json_dumps([float(summary["max_power"]) for summary in history_summaries]),
                            "past_beam_entropy": _json_dumps(
                                [float(summary["beam_entropy"]) for summary in history_summaries]
                            ),
                            "noisy_position_history": _json_dumps(noisy_hist.tolist()),
                            "clean_position_history": _json_dumps(clean_hist.tolist()),
                            "label_beam_future": int(beam_labels[0]),
                            "label_blockage_future": int(blockage[0]),
                            "label_trajectory_future": _json_dumps(trajectory.tolist()),
                            "los_status_future": _json_dumps(los_future.tolist()),
                            "los_status_source_future": _json_dumps(
                                [str(summary["los_status_source"]) for summary in future_summaries]
                            ),
                            "link_state_future": _json_dumps(link_state.tolist()),
                            "blockage_labels_future": _json_dumps(blockage.tolist()),
                            "blockage_valid_mask": _json_dumps(blockage_valid_mask.tolist()),
                            "beam_gain_future": _json_dumps(gain_future.tolist()),
                            "valid_mask": _json_dumps([True] * self.pred_horizon),
                            "split": "",
                        }
                    )
                    rows.append(row)
                    sample_id = str(row["sample_id"])
                    labels_acc["sample_id"].append(sample_id)
                    labels_acc["beam_label"].append(int(beam_labels[0]))
                    labels_acc["beam_labels_future"].append(beam_labels)
                    labels_acc["blockage_label"].append(int(blockage[0]))
                    labels_acc["blockage_labels_future"].append(blockage)
                    labels_acc["blockage_valid_mask"].append(blockage_valid_mask)
                    labels_acc["trajectory_future"].append(trajectory)
                    labels_acc["los_status_future"].append(los_future)
                    labels_acc["link_state_future"].append(link_state)
                    labels_acc["beam_gain_future"].append(gain_future)
                    labels_acc["valid_mask"].append(np.ones(self.pred_horizon, dtype=bool))
                    weak_wireless_history.append(weak_hist)
                    if self.enable_radar:
                        radar_feature_history.append(radar_hist)
                    noisy_position_history.append(noisy_hist.astype(np.float32))
                    clean_position_history.append(clean_hist.astype(np.float32))

        return {
            "rows": rows,
            "labels": _stack_label_arrays(labels_acc, self.pred_horizon, self.num_beams),
            "weak_wireless": {
                "sample_id": np.asarray(labels_acc["sample_id"], dtype=str),
                "weak_wireless_history": _stack_or_empty(weak_wireless_history, (0, self.seq_len, 4), np.float32),
            },
            "radar_features": {
                "sample_id": np.asarray(labels_acc["sample_id"], dtype=str),
                "radar_feature_history": _stack_or_empty(radar_feature_history, (0, self.seq_len, RADAR_FEATURE_SIZE), np.float32),
            },
            "noisy_position": {
                "sample_id": np.asarray(labels_acc["sample_id"], dtype=str),
                "noisy_position_history": _stack_or_empty(noisy_position_history, (0, self.seq_len, 2), np.float32),
                "clean_position_history": _stack_or_empty(clean_position_history, (0, self.seq_len, 2), np.float32),
            },
            "skip_counts": dict(self.skip_counts),
            "los_status_source_counts": dict(self.los_status_source_counts),
        }


    def write_cache(
        self,
        output_root: str | Path,
        *,
        split_by: str = "sequence",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
    ) -> dict[str, Any]:
        return write_label_cache(
            self,
            output_root,
            split_by=split_by,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    def _resolve_ue_ids(self) -> list[int]:
        if self.ue_ids is not None:
            return [int(value) for value in self.ue_ids]
        for attr in ("ue_ids", "ue_indices", "user_ids"):
            values = getattr(self.dataset, attr, None)
            if values is not None:
                return [int(value) for value in values]
        mobility_dataset = getattr(self.dataset, "mobility_dataset", None)
        objects = getattr(mobility_dataset, "objects", None)
        if isinstance(objects, Mapping):
            return [int(value) for value in sorted(objects)]
        try:
            mobility = self.dataset.get_sample("mobility")
        except Exception:
            mobility = None
        if isinstance(mobility, Mapping):
            return [int(value) for value in sorted(mobility)]
        return [0]

    def _mobility_for_ue(self, ue_id: int) -> MobilityTrace:
        sample = _get_sample(
            self.dataset,
            ["mobility", "mobility-ue"],
            [{"object_id": ue_id}, {"ue_idx": ue_id}, {"ue_id": ue_id}, {}],
        )
        if isinstance(sample, Mapping) and ue_id in sample:
            sample = sample[ue_id]
        if isinstance(sample, Mapping) and str(ue_id) in sample:
            sample = sample[str(ue_id)]
        info = sample.get_all_samples() if hasattr(sample, "get_all_samples") else sample
        locations = _field(info, "location", "locations", "position", "positions")
        if locations is None:
            raise KeyError("mobility sample does not contain location.")
        locations_array = np.asarray(locations, dtype=np.float32)
        if locations_array.ndim != 2 or locations_array.shape[1] < 2:
            raise ValueError(f"mobility location must have shape [T, >=2], got {locations_array.shape}.")
        times = _field(info, "time", "times")
        times_array = np.asarray(times if times is not None else np.arange(len(locations_array)))
        if len(times_array) != len(locations_array):
            raise ValueError("mobility time and location lengths do not match.")
        return MobilityTrace(times=times_array, locations=locations_array, info=info)

    def _comm_summary(self, time_value: Any, local_idx: int, *, bs_id: int, ue_id: int) -> dict[str, Any]:
        sample = _get_sample(
            self.dataset,
            ["comm-ue", "comm", "communication"],
            [
                {"index": time_value, "bs_idx": bs_id, "ue_idx": ue_id},
                {"index": local_idx, "bs_idx": bs_id, "ue_idx": ue_id},
                {"time": time_value, "bs_idx": bs_id, "ue_idx": ue_id},
                {"index": time_value, "bs_id": bs_id, "ue_id": ue_id},
                {"index": local_idx, "bs_id": bs_id, "ue_id": ue_id},
            ],
        )
        channel = _field(sample, "coeffs", "channel", "channels", "H", "h")
        if channel is None:
            raise KeyError("comm sample does not contain channel coefficients.")
        channel_array = np.asarray(channel)
        codebook = make_ula_dft_codebook(_infer_num_ant(channel_array), self.num_beams)
        gain = compute_beam_gain(channel_array, codebook)
        beam_label = int(np.argmax(gain))
        topk = min(self.beam_topk, len(gain))
        topk_indices = np.argsort(gain)[-topk:][::-1].astype(np.int64)
        topk_powers = gain[topk_indices].astype(np.float32)
        max_power = float(gain[beam_label])
        margin = float(topk_powers[0] - topk_powers[1]) if len(topk_powers) > 1 else max_power
        entropy = beam_entropy(gain)
        los_status, los_status_source = self._resolve_los_status(
            sample,
            time_value=time_value,
            local_idx=local_idx,
            bs_id=bs_id,
            ue_id=ue_id,
        )
        if los_status is None:
            raise KeyError("comm sample does not contain LoS_status.")
        self.los_status_source_counts[los_status_source] += 1
        weak = np.asarray(
            [
                beam_label / float(self.num_beams - 1),
                max_power,
                margin,
                entropy,
            ],
            dtype=np.float32,
        )
        return {
            "beam_label": beam_label,
            "beam_gain": gain.astype(np.float32),
            "topk_beam_indices": topk_indices,
            "topk_beam_powers": topk_powers,
            "max_power": max_power,
            "beam_entropy": entropy,
            "los_status": int(los_status),
            "los_status_source": los_status_source,
            "weak_wireless": weak,
        }

    def _resolve_los_status(
        self,
        sample: Any,
        *,
        time_value: Any,
        local_idx: int,
        bs_id: int,
        ue_id: int,
    ) -> tuple[int | None, str]:
        raw_los = self._raw_los_status_for_time(time_value, local_idx, bs_id=bs_id, ue_id=ue_id)
        if raw_los is not None:
            return raw_los, "raw_raytracing_mat"

        los_status = _field(sample, "LoS_status", "los_status", "LOS_status", "los")
        if los_status is None:
            return None, "missing"
        return int(np.asarray(los_status).reshape(-1)[0]), "sample_field"

    def _raw_los_status_for_time(self, time_value: Any, local_idx: int, *, bs_id: int, ue_id: int) -> int | None:
        scenario_root = self._resolved_scenario_root()
        if scenario_root is None:
            return None
        scene_id = self._scene_id_for_time(time_value, local_idx)
        if scene_id is None:
            return None
        cache_key = (int(scene_id), int(bs_id), int(ue_id))
        if cache_key not in self._raw_los_cache:
            self._raw_los_cache[cache_key] = self._read_raw_los_status(
                scenario_root,
                scene_id=int(scene_id),
                bs_id=int(bs_id),
                ue_id=int(ue_id),
            )
        return self._raw_los_cache[cache_key]

    def _resolved_scenario_root(self) -> Path | None:
        if self.scenario_root is not None:
            return Path(self.scenario_root).expanduser().resolve()
        params = getattr(self.dataset, "params", None)
        if isinstance(params, Mapping):
            root = params.get("dataset_folder")
            if root is not None:
                return Path(root).expanduser().resolve()
        return None

    def _scene_id_for_time(self, time_value: Any, local_idx: int) -> int | None:
        if self.scenes is not None and len(self.scenes) > local_idx:
            return int(self.scenes[local_idx])
        try:
            array = np.asarray(time_value)
            if array.ndim == 0:
                return int(array.item())
        except Exception:
            return None
        return None

    def _read_raw_los_status(self, scenario_root: Path, *, scene_id: int, bs_id: int, ue_id: int) -> int | None:
        located = self._locate_bs_ue_mat(scenario_root, bs_id=bs_id, ue_id=ue_id)
        if located is None:
            return None
        mat_path, ue_offset = located
        scene_mat_path = scenario_root / self.scenario / "wireless" / f"scene_{scene_id}" / mat_path.name
        if not scene_mat_path.exists():
            return None
        try:
            import scipy.io

            data = scipy.io.loadmat(scene_mat_path, variable_names=["channels"])
            channels = data["channels"]
            record = channels[0][ue_offset][0][0]
            path_matrix = record["p"]
            return _link_los_from_path_statuses(path_matrix[7])
        except Exception:
            return None

    def _locate_bs_ue_mat(self, scenario_root: Path, *, bs_id: int, ue_id: int) -> tuple[Path, int] | None:
        cache_key = (int(bs_id), int(ue_id))
        if cache_key in self._bs_ue_file_cache:
            return self._bs_ue_file_cache[cache_key]

        wireless_root = scenario_root / self.scenario / "wireless"
        candidates = sorted(wireless_root.glob(f"scene_*/BS{int(bs_id) + 1}_UE_*-*.mat"))
        for path in candidates:
            parsed = _parse_ue_file_range(path.name)
            if parsed is None:
                continue
            start, end = parsed
            if start <= int(ue_id) <= end:
                located = (path, int(ue_id) - start)
                self._bs_ue_file_cache[cache_key] = located
                return located

        self._bs_ue_file_cache[cache_key] = None
        return None

    def _paths_for_time(self, kind: str, time_value: Any, local_idx: int) -> list[str]:
        names = {
            "camera": ["cam", "camera", "rgb"],
            "lidar": ["lidar", "lidar-ue"],
        }[kind]
        device_ids = self.camera_ids if kind == "camera" else self.lidar_ids
        paths: list[str] = []
        for device_id in device_ids:
            for candidate_id in _device_id_candidates(device_id):
                sample = _get_sample(
                    self.dataset,
                    names,
                    [
                        {"index": time_value, "device_index": candidate_id},
                        {"index": local_idx, "device_index": candidate_id},
                        {"time": time_value, "device_index": candidate_id},
                        {"index": time_value, "device_id": candidate_id},
                        {"index": local_idx, "device_id": candidate_id},
                        {"index": time_value},
                        {"index": local_idx},
                    ],
                    default=None,
                )
                extracted = _extract_paths(sample)
                if extracted:
                    paths.extend(extracted)
                    break
        return sorted(set(paths))

    def _radar_feature_for_time(self, time_value: Any, local_idx: int, *, bs_id: int, ue_id: int) -> np.ndarray:
        sample = _get_sample(
            self.dataset,
            ["radar"],
            [
                {"index": time_value, "bs_idx": bs_id, "ue_idx": ue_id},
                {"index": local_idx, "bs_idx": bs_id, "ue_idx": ue_id},
                {"time": time_value, "bs_idx": bs_id, "ue_idx": ue_id},
            ],
        )
        return extract_radar_feature(sample)



__all__ = [
    "BLOCKAGE_IGNORE_INDEX",
    "DeepVerseLabelBuilder",
    "LINK_STATE_LOS",
    "LINK_STATE_NLOS",
    "LINK_STATE_UNKNOWN",
    "MobilityTrace",
    "extract_radar_feature",
    "los_to_blockage",
    "los_to_link_state",
]
