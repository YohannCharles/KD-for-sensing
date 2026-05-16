from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .codebook import beam_entropy, compute_beam_gain, make_ula_dft_codebook
from .sanity_check import build_sanity_report
from .split import assign_splits, make_split_result


BLOCKAGE_IGNORE_INDEX = -100
LINK_STATE_UNKNOWN = -1
LINK_STATE_LOS = 0
LINK_STATE_NLOS = 1


@dataclass(frozen=True)
class MobilityTrace:
    times: np.ndarray
    locations: np.ndarray
    info: Any


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
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        built = self.build()
        rows = list(built["rows"])
        split_result = make_split_result(
            rows,
            split_by=split_by,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=self.seed,
        )
        split = split_result.split
        if split_result.discarded_sample_ids:
            keep_ids = {sample_id for sample_ids in split.values() for sample_id in sample_ids}
            rows = [row for row in rows if str(row["sample_id"]) in keep_ids]
            built = _filter_built_by_sample_ids(built, keep_ids)
        assign_splits(rows, split)

        paths = {
            "metadata": output_path / "metadata.json",
            "samples": output_path / "samples.csv",
            "labels": output_path / "labels.npz",
            "weak_wireless": output_path / "weak_wireless.npz",
            "radar_features": output_path / "radar_features.npz",
            "noisy_position": output_path / "noisy_position.npz",
            "camera_index": output_path / "camera_index.json",
            "lidar_index": output_path / "lidar_index.json",
            "split": output_path / "split.json",
            "sanity_report": output_path / "sanity_report.json",
        }

        pd.DataFrame(rows).to_csv(paths["samples"], index=False)
        np.savez_compressed(paths["labels"], **built["labels"])
        np.savez_compressed(paths["weak_wireless"], **built["weak_wireless"])
        np.savez_compressed(paths["radar_features"], **built["radar_features"])
        np.savez_compressed(paths["noisy_position"], **built["noisy_position"])

        _write_json(paths["camera_index"], _path_index(rows, "camera_paths"))
        _write_json(paths["lidar_index"], _path_index(rows, "lidar_paths"))
        _write_json(paths["split"], split)

        blockage = _blockage_metadata(
            built["labels"],
            min_class_count=self.blockage_min_class_count,
            min_class_ratio=self.blockage_min_class_ratio,
        )
        report = build_sanity_report(
            rows=rows,
            labels=built["labels"],
            split=split,
            skip_counts=built["skip_counts"],
            artifact_paths={key: str(path) for key, path in paths.items()},
            radar_features=built["radar_features"].get("radar_feature_history"),
            split_metadata=split_result.metadata,
            blockage=blockage,
        )
        default_inputs = ["camera", "lidar", "weak_wireless", "noisy_position"]
        if self.enable_radar:
            default_inputs.insert(2, "radar")
        default_objectives = ["beam", "trajectory"]
        if blockage["usable"]:
            default_objectives.append("blockage")
        metadata = {
            "scenario": self.scenario,
            "seq_len": self.seq_len,
            "pred_horizon": self.pred_horizon,
            "num_beams": self.num_beams,
            "beam_topk": self.beam_topk,
            "position_noise_std": self.position_noise_std,
            "seed": self.seed,
            "split_by": split_result.metadata["effective_split_by"],
            "requested_split_by": split_result.metadata["requested_split_by"],
            "split_protocol": split_result.metadata,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "sample_count": len(rows),
            "skip_counts": built["skip_counts"],
            "los_status_source_counts": _los_status_source_counts(rows),
            "split_counts": {name: len(sample_ids) for name, sample_ids in split.items()},
            "label_distribution": report["label_distribution"],
            "blockage": blockage,
            "default_inputs": default_inputs,
            "default_objectives": default_objectives,
            "radar_feature_size": RADAR_FEATURE_SIZE if self.enable_radar else 0,
            "radar_feature_names": RADAR_FEATURE_NAMES if self.enable_radar else [],
            "oracle_only_fields": ["clean_position_history", "beam_gain_future", "los_status_future"],
            "artifacts": {key: str(path) for key, path in paths.items()},
        }
        _write_json(paths["metadata"], metadata)
        _write_json(paths["sanity_report"], report)

        return {
            "paths": {key: str(path) for key, path in paths.items()},
            "metadata": metadata,
            "sanity_report": report,
        }

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


def los_to_link_state(los_status: int) -> int:
    status = int(los_status)
    if status == 1:
        return LINK_STATE_LOS
    if status == 0:
        return LINK_STATE_NLOS
    return LINK_STATE_UNKNOWN


def los_to_blockage(los_status: int) -> int:
    link_state = los_to_link_state(los_status)
    if link_state == LINK_STATE_LOS:
        return 0
    if link_state == LINK_STATE_NLOS:
        return 1
    return BLOCKAGE_IGNORE_INDEX


def _link_los_from_path_statuses(path_statuses: Any) -> int:
    statuses = np.asarray(path_statuses).reshape(-1)
    finite_statuses = statuses[np.isfinite(statuses.astype(np.float64, copy=False))]
    if finite_statuses.size == 0:
        return LINK_STATE_UNKNOWN
    if np.any(np.isclose(finite_statuses, 1.0)):
        return 1
    if np.any(np.isclose(finite_statuses, 0.0)):
        return 0
    return LINK_STATE_UNKNOWN


def _parse_ue_file_range(filename: str) -> tuple[int, int] | None:
    stem = Path(filename).stem
    if "_UE_" not in stem:
        return None
    _, range_part = stem.rsplit("_UE_", 1)
    if "-" not in range_part:
        return None
    start_text, end_text = range_part.split("-", 1)
    try:
        return int(start_text), int(end_text)
    except ValueError:
        return None


RADAR_FEATURE_NAMES = [
    "abs_mean",
    "abs_std",
    "abs_max",
    "phase_diff_mean",
    "phase_diff_std",
    "path_count",
]
RADAR_FEATURE_SIZE = len(RADAR_FEATURE_NAMES)


def extract_radar_feature(radar_sample: Any) -> np.ndarray:
    coeffs = _field(radar_sample, "coeffs", "channel", "channels", "H", "h", "tensor", "data")
    if coeffs is None:
        raise KeyError("radar sample does not contain coefficients or tensor data.")
    array = np.asarray(coeffs)
    if array.size == 0:
        raise ValueError("radar coefficients are empty.")
    magnitude = np.abs(array.astype(np.complex64, copy=False)).astype(np.float32)
    phase = np.unwrap(np.angle(array.reshape(-1).astype(np.complex64, copy=False)))
    phase_diff = np.diff(phase).astype(np.float32)
    if phase_diff.size == 0:
        phase_diff = np.zeros(1, dtype=np.float32)
    features = np.asarray(
        [
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.max(magnitude)),
            float(np.mean(phase_diff)),
            float(np.std(phase_diff)),
            float(_path_count(_field(radar_sample, "paths", "ray_paths", "rays"))),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("radar feature contains NaN or Inf.")
    return features


def _get_sample(dataset: Any, names: Sequence[str], kwarg_sets: Sequence[dict[str, Any]], default: Any = ...):
    get_sample = getattr(dataset, "get_sample")
    last_exc: Exception | None = None
    for name in names:
        for kwargs in kwarg_sets:
            try:
                return get_sample(name, **kwargs)
            except Exception as exc:
                last_exc = exc
    if default is not ...:
        return default
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No get_sample candidates were attempted.")


def _field(obj: Any, *names: str) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_paths(sample: Any) -> list[str]:
    if sample is None:
        return []
    if isinstance(sample, (str, Path)):
        return [str(sample)]
    if isinstance(sample, Mapping):
        for key in ("path", "paths", "filepath", "file_path", "image_path", "lidar_path", "filename"):
            if key in sample:
                return _extract_paths(sample[key])
    if isinstance(sample, Sequence) and not isinstance(sample, (bytes, bytearray)):
        paths: list[str] = []
        for item in sample:
            paths.extend(_extract_paths(item))
        return paths
    for key in ("path", "paths", "filepath", "file_path", "image_path", "lidar_path", "filename"):
        if hasattr(sample, key):
            return _extract_paths(getattr(sample, key))
    return []


def _path_count(paths: Any) -> int:
    if paths is None:
        return 0
    if isinstance(paths, Mapping):
        if "paths" in paths:
            return _path_count(paths["paths"])
        for value in paths.values():
            try:
                return len(value)
            except TypeError:
                continue
        return len(paths)
    for attr in ("num_paths", "n_paths"):
        if hasattr(paths, attr):
            value = getattr(paths, attr)
            return int(value() if callable(value) else value)
    for attr in ("paths", "ToA", "DoD_theta", "DoA_theta", "phase"):
        if hasattr(paths, attr):
            try:
                return len(getattr(paths, attr))
            except TypeError:
                continue
    try:
        return len(paths)
    except TypeError:
        return 0


def _infer_num_ant(channel: np.ndarray) -> int:
    if channel.ndim == 0:
        raise ValueError("channel coefficients are scalar.")
    if channel.ndim == 1:
        return int(channel.shape[0])
    if channel.ndim == 2:
        return int(channel.shape[0])
    first = int(channel.shape[0])
    second = int(channel.shape[1])
    if first == 1 and second > 1:
        return second
    return first


def _device_id_candidates(device_id: Any) -> list[Any]:
    candidates = [device_id]
    if isinstance(device_id, (int, np.integer)) and int(device_id) > 0:
        candidates.append(int(device_id) - 1)
    return candidates


def _metadata_value(info: Any, local_idx: int, names: Sequence[str], *, default: Any) -> Any:
    for name in names:
        value = _field(info, name)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            return value.decode() if isinstance(value, bytes) else value
        try:
            array = np.asarray(value)
        except Exception:
            return value
        if array.ndim == 0:
            return array.item()
        if len(array) > local_idx:
            return array[local_idx].item() if np.asarray(array[local_idx]).ndim == 0 else array[local_idx]
        return value
    return default


def _dataset_value(dataset: Any, names: Sequence[str], default: Any) -> Any:
    for name in names:
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            return value() if callable(value) else value
    return default


def _make_group_key(scene_id: Any, sequence_id: Any, segment_id: Any, object_id: Any) -> str:
    return "|".join(
        [
            f"scene={_id_part(scene_id)}",
            f"sequence={_id_part(sequence_id)}",
            f"segment={_id_part(segment_id)}",
            f"object={_id_part(object_id)}",
        ]
    )


def _make_sample_id(
    scenario: str,
    *,
    scene_id: Any,
    sequence_id: Any,
    segment_id: Any,
    object_id: Any,
    ue_id: Any,
    bs_id: Any,
    t_anchor: Any,
) -> str:
    return "_".join(
        [
            _id_part(scenario),
            f"scene{_id_part(scene_id)}",
            f"seq{_id_part(sequence_id) or 'single'}",
            f"seg{_id_part(segment_id) or 'full'}",
            f"obj{_id_part(object_id)}",
            f"ue{_id_part(ue_id)}",
            f"bs{_id_part(bs_id)}",
            f"t{_time_to_id(t_anchor)}",
        ]
    )


def _id_part(value: Any) -> str:
    scalar = _json_scalar(value)
    text = str(scalar)
    return (
        text.replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("|", "-")
        .replace(":", "-")
        .replace(".", "p")
        .replace("-", "m")
    )


def _filter_built_by_sample_ids(built: dict[str, Any], keep_ids: set[str]) -> dict[str, Any]:
    labels = built["labels"]
    mask = np.asarray([str(sample_id) in keep_ids for sample_id in labels["sample_id"]], dtype=bool)
    filtered = dict(built)
    filtered["labels"] = _filter_array_dict(labels, mask)
    for key in ("weak_wireless", "radar_features", "noisy_position"):
        filtered[key] = _filter_array_dict(built[key], mask)
    return filtered


def _filter_array_dict(payload: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.shape[:1] == mask.shape:
            filtered[key] = array[mask]
        else:
            filtered[key] = array
    return filtered


def _blockage_metadata(labels: dict[str, np.ndarray], *, min_class_count: int, min_class_ratio: float) -> dict[str, Any]:
    raw_los = labels.get("los_status_future", np.asarray([], dtype=np.int16))
    blockage = labels.get("blockage_labels_future", np.asarray([], dtype=np.int64))
    valid_mask = labels.get("blockage_valid_mask", np.asarray([], dtype=bool)).astype(bool)
    valid_labels = blockage[valid_mask] if blockage.shape == valid_mask.shape else np.asarray([], dtype=np.int64)
    distribution = _counter_dict(valid_labels)
    raw_distribution = _counter_dict(raw_los)
    total = int(valid_labels.size)
    class_counts = {label: int(np.sum(valid_labels == label)) for label in (0, 1)}
    present_classes = {label for label, count in class_counts.items() if count > 0}
    minority_count = min(class_counts.values()) if total else 0
    minority_ratio = (minority_count / total) if total else 0.0

    reason = ""
    usable = True
    if total == 0:
        usable = False
        reason = "no_valid_blockage_labels"
    elif present_classes != {0, 1}:
        usable = False
        missing = sorted({0, 1} - present_classes)
        reason = f"missing_classes:{','.join(str(value) for value in missing)}"
    elif minority_count < min_class_count:
        usable = False
        reason = "minority_class_count_below_min"
    elif minority_ratio < min_class_ratio:
        usable = False
        reason = "minority_class_ratio_below_min"

    return {
        "usable": usable,
        "reason": reason,
        "ignore_index": BLOCKAGE_IGNORE_INDEX,
        "min_class_count": int(min_class_count),
        "min_class_ratio": float(min_class_ratio),
        "raw_los_status_distribution": raw_distribution,
        "valid_label_distribution": distribution,
        "valid_label_count": total,
        "minority_class_count": int(minority_count),
        "minority_class_ratio": float(minority_ratio),
    }


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    flat = np.asarray(values).reshape(-1)
    return {str(k): int(v) for k, v in Counter(flat.tolist()).items()}


def _stack_label_arrays(labels: dict[str, list[Any]], pred_horizon: int, num_beams: int) -> dict[str, np.ndarray]:
    sample_ids = np.asarray(labels["sample_id"], dtype=str)
    count = len(sample_ids)
    return {
        "sample_id": sample_ids,
        "beam_label": np.asarray(labels["beam_label"], dtype=np.int64),
        "beam_labels_future": _stack_or_empty(labels["beam_labels_future"], (count, pred_horizon), np.int64),
        "blockage_label": np.asarray(labels["blockage_label"], dtype=np.int64),
        "blockage_labels_future": _stack_or_empty(labels["blockage_labels_future"], (count, pred_horizon), np.int64),
        "blockage_valid_mask": _stack_or_empty(labels["blockage_valid_mask"], (count, pred_horizon), bool),
        "trajectory_future": _stack_or_empty(labels["trajectory_future"], (count, pred_horizon, 2), np.float32),
        "los_status_future": _stack_or_empty(labels["los_status_future"], (count, pred_horizon), np.int16),
        "link_state_future": _stack_or_empty(labels["link_state_future"], (count, pred_horizon), np.int16),
        "beam_gain_future": _stack_or_empty(labels["beam_gain_future"], (count, pred_horizon, num_beams), np.float32),
        "valid_mask": _stack_or_empty(labels["valid_mask"], (count, pred_horizon), bool),
    }


def _stack_or_empty(values: Sequence[Any], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    if values:
        return np.stack(values).astype(dtype)
    return np.empty(shape, dtype=dtype)


def _path_index(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    return {str(row["sample_id"]): json.loads(row[column]) for row in rows}


def _los_status_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        payload = row.get("los_status_source_future", "[]")
        try:
            values = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            values = []
        if isinstance(values, list):
            counts.update(str(value) for value in values)
    return dict(counts)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_dumps(payload: Any) -> str:
    return json.dumps(_jsonable(payload), separators=(",", ":"))


def _json_scalar(value: Any) -> int | float | str:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (np.integer, int)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item)
        return str(item)
    return str(value)


def _time_to_id(value: Any) -> str:
    scalar = _json_scalar(value)
    if isinstance(scalar, float) and scalar.is_integer():
        scalar = int(scalar)
    return str(scalar).replace(".", "p").replace("-", "m")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
