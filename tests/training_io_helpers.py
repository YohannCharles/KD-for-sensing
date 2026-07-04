from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset

class _TinyImageBatchModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3, 0.0], dtype=torch.float32))
        self.num_classes = num_classes
        self.calls = 0

    def forward(self, image_batch=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        batch_size = image_batch.shape[0]
        horizon = 1
        logits = self.weight.view(1, 1, self.num_classes).expand(batch_size, horizon, self.num_classes)
        features = logits.detach().clone()
        return {"logits": logits, "input_features": features, "output_features": features}

class _DisabledGradScaler:
    def is_enabled(self) -> bool:
        return False

def _removed_image_option(suffix: str) -> str:
    return "image_" + "motion_" + suffix

def _removed_image_profile() -> str:
    return "motion" + "_mask"

def _removed_encoder_name(prefix: str = "") -> str:
    return prefix + "motion" + "_cnn"

def _seq_index_keys_for_dataset(dataset) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for leaf, indices in _leaf_indices_for_test(dataset):
        frame = pd.read_csv(leaf.root_csv, na_values="").fillna(-99)
        for idx in indices:
            keys.add((str(getattr(leaf, "scene_id", "")), str(frame.iloc[int(idx)]["seq_index"])))
    return keys

def _leaf_indices_for_test(dataset) -> list[tuple[object, list[int]]]:
    if isinstance(dataset, ConcatDataset):
        result: list[tuple[object, list[int]]] = []
        for component in dataset.datasets:
            result.extend(_leaf_indices_for_test(component))
        return result
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if isinstance(parent, ConcatDataset):
            grouped: dict[int, list[int]] = {}
            cumulative = list(parent.cumulative_sizes)
            for raw_index in dataset.indices:
                global_index = int(raw_index)
                component_idx = int(np.searchsorted(cumulative, global_index, side="right"))
                previous = cumulative[component_idx - 1] if component_idx > 0 else 0
                grouped.setdefault(component_idx, []).append(global_index - previous)
            result: list[tuple[object, list[int]]] = []
            for component_idx, local_indices in sorted(grouped.items()):
                component = parent.datasets[component_idx]
                if isinstance(component, Subset):
                    base_pairs = _leaf_indices_for_test(component)
                    if len(base_pairs) == 1:
                        base_dataset, base_indices = base_pairs[0]
                        result.append((base_dataset, [base_indices[int(index)] for index in local_indices]))
                    else:
                        result.extend(base_pairs)
                else:
                    result.append((component, [int(index) for index in local_indices]))
            return result
        if isinstance(parent, Subset):
            base_pairs = _leaf_indices_for_test(parent)
            if len(base_pairs) == 1:
                base_dataset, base_indices = base_pairs[0]
                return [(base_dataset, [base_indices[int(index)] for index in dataset.indices])]
        return [(parent, [int(index) for index in dataset.indices])]
    return [(dataset, list(range(len(dataset))))]

def _write_full_sequence_fixture(root: Path, csv_path: Path, *, seq_len: int, num_pred: int) -> None:
    for idx in range(seq_len):
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
        np.savetxt(root / f"gps_{idx}.txt", np.asarray([42.0 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / f"bs_gps_{idx}.txt", np.asarray([42.0, -71.0], dtype=np.float32))
    for idx in range(num_pred):
        future = np.zeros(64, dtype=np.float32)
        future[idx + 10] = 1.0
        np.savetxt(root / f"future_{idx}.txt", future)
    columns = (
        [f"camera{i}" for i in range(1, seq_len + 1)]
        + [f"radar{i}" for i in range(1, seq_len + 1)]
        + [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"lidar{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(seq_len)]
        + [f"radar_{idx}_RA.npy" for idx in range(seq_len)]
        + [f"gps_{idx}.txt" for idx in range(seq_len)]
        + [f"bs_gps_{idx}.txt" for idx in range(seq_len)]
        + [f"lidar_{idx}.txt" for idx in range(seq_len)]
        + [f"beam_{idx}.txt" for idx in range(seq_len)]
        + [f"future_{idx}.txt" for idx in range(num_pred)]
        + ["1"]
    )
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")

def _write_multirow_gps_sequence_fixture(root: Path, csv_path: Path, *, rows: int, seq_len: int, num_pred: int) -> None:
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    lines = [",".join(columns)]
    for row_idx in range(rows):
        gps_paths = []
        bs_paths = []
        beam_paths = []
        future_paths = []
        for idx in range(seq_len):
            gps_name = f"row{row_idx}_gps_{idx}.txt"
            bs_name = f"row{row_idx}_bs_{idx}.txt"
            beam_name = f"row{row_idx}_beam_{idx}.txt"
            np.savetxt(root / gps_name, np.asarray([42.0 + row_idx * 1e-4 + idx * 1e-5, -71.0], dtype=np.float32))
            np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
            beam = np.zeros(64, dtype=np.float32)
            beam[(row_idx + idx) % 64] = 1.0
            np.savetxt(root / beam_name, beam)
            gps_paths.append(gps_name)
            bs_paths.append(bs_name)
            beam_paths.append(beam_name)
        for idx in range(num_pred):
            future_name = f"row{row_idx}_future_{idx}.txt"
            future = np.zeros(64, dtype=np.float32)
            future[(row_idx + idx + 10) % 64] = 1.0
            np.savetxt(root / future_name, future)
            future_paths.append(future_name)
        values = gps_paths + bs_paths + beam_paths + future_paths + [str(row_idx)]
        lines.append(",".join(values))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_stratified_gps_sequence_fixture(
    root: Path,
    csv_path: Path,
    *,
    rows: int,
    offset: int,
    seq_block_size: int = 1,
) -> None:
    seq_len = 5
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + ["future_beam1", "seq_index"]
    )
    lines = [",".join(columns)]
    for row_idx in range(rows):
        global_idx = int(offset + row_idx)
        label = 10 if row_idx % 2 == 0 else 20
        gps_paths = []
        bs_paths = []
        beam_paths = []
        for frame_idx in range(seq_len):
            gps_name = f"s{offset}_row{row_idx}_gps_{frame_idx}.txt"
            bs_name = f"s{offset}_row{row_idx}_bs_{frame_idx}.txt"
            beam_name = f"s{offset}_row{row_idx}_beam_{frame_idx}.txt"
            np.savetxt(root / gps_name, np.asarray([42.0 + global_idx * 1e-4 + frame_idx * 1e-5, -71.0]))
            np.savetxt(root / bs_name, np.asarray([42.0, -71.0]))
            beam = np.zeros(64, dtype=np.float32)
            beam[(label + frame_idx) % 64] = 1.0
            np.savetxt(root / beam_name, beam)
            gps_paths.append(gps_name)
            bs_paths.append(bs_name)
            beam_paths.append(beam_name)
        future_name = f"s{offset}_row{row_idx}_future.txt"
        future = np.zeros(64, dtype=np.float32)
        future[label] = 1.0
        np.savetxt(root / future_name, future)
        seq_index = int(offset + (row_idx // max(int(seq_block_size), 1)))
        lines.append(",".join(gps_paths + bs_paths + beam_paths + [future_name, str(seq_index)]))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_aux_training_csv(root: Path, csv_path: Path, *, prefix: str, future_max: list[float]) -> None:
    seq_len = 2
    num_pred = len(future_max)
    gps_paths = []
    bs_paths = []
    beam_paths = []
    future_paths = []
    future_gps_paths = []
    future_bs_paths = []
    for idx in range(seq_len):
        gps_name = f"{prefix}_gps_{idx}.txt"
        bs_name = f"{prefix}_bs_{idx}.txt"
        beam_name = f"{prefix}_beam_{idx}.txt"
        np.savetxt(root / gps_name, np.asarray([42.0 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / beam_name, beam)
        gps_paths.append(gps_name)
        bs_paths.append(bs_name)
        beam_paths.append(beam_name)
    for idx, max_power in enumerate(future_max):
        future_name = f"{prefix}_future_{idx}.txt"
        gps_name = f"{prefix}_future_gps_{idx}.txt"
        bs_name = f"{prefix}_future_bs_{idx}.txt"
        future = np.linspace(0.1, float(max_power), 64, dtype=np.float32)
        np.savetxt(root / future_name, future)
        np.savetxt(root / gps_name, np.asarray([42.0001 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        future_paths.append(future_name)
        future_gps_paths.append(gps_name)
        future_bs_paths.append(bs_name)
    columns = (
        [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + [f"future_beam{i}" for i in range(1, num_pred + 1)]
        + [f"future_gps{i}" for i in range(1, num_pred + 1)]
        + [f"future_bs_gps{i}" for i in range(1, num_pred + 1)]
        + ["seq_index"]
    )
    values = gps_paths + bs_paths + beam_paths + future_paths + future_gps_paths + future_bs_paths + ["1"]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")

def _write_camera_files(root: Path, *, count: int) -> None:
    from PIL import Image

    for idx in range(count):
        Image.fromarray(np.full((8, 8, 3), idx * 40, dtype=np.uint8)).save(root / f"camera_{idx}.jpg")

def _write_minimal_csv(path: Path, *, camera: bool, radar: bool, gps: bool, lidar: bool) -> None:
    columns: list[str] = []
    values: list[str] = []
    if camera:
        columns.append("camera1")
        values.append("camera.jpg")
    if radar:
        columns.append("radar1")
        values.append("radar_RA.npy")
    if gps:
        columns.extend(["gps1", "bs_gps1"])
        values.extend(["gps.txt", "bs_gps.txt"])
    if lidar:
        columns.append("lidar1")
        values.append("lidar.txt")
    columns.extend(["beam1", "future_beam1", "seq_index"])
    values.extend(["beam.txt", "future.txt", "1"])
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")
