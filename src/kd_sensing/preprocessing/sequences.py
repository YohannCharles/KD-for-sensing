from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from kd_sensing.data.split_metadata import (
    SPLIT_METADATA_PROTOCOL,
    SUPPORTED_SPLIT_METADATA_PROTOCOLS,
    default_split_metadata_path,
)
from kd_sensing.preprocessing.sequence_metadata import (
    label_distribution_summary,
    write_split_metadata,
)
from kd_sensing.preprocessing.sequence_splits import (
    SequenceSplit,
    SplitProtocolPlan,
    resolve_split_protocol,
    select_balanced_sequence_split,
)
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


@dataclass(frozen=True)
class SequenceColumnPlan:
    base_columns: list[str]
    gps_source_column: str
    bs_gps_column: str
    lidar_source_column: str
    mmwave_source_column: str


def resolve_sequence_column_plan(
    all_data: pd.DataFrame,
    *,
    csv_path: Path,
    include_gps: bool,
    gps_column: str,
    gps_fallback_column: str | None,
    bs_gps_column: str,
    include_lidar: bool,
    lidar_column: str,
    lidar_fallback_column: str | None,
    include_mmwave: bool,
    mmwave_column: str,
    mmwave_fallback_column: str | None,
    include_position_targets: bool,
) -> SequenceColumnPlan:
    base_columns = ["unit1_rgb", "unit1_radar", "unit1_pwr_60ghz", "seq_index"]
    require_columns(all_data, csv_path, base_columns)
    gps_source_column = gps_column
    lidar_source_column = lidar_column
    mmwave_source_column = mmwave_column
    if include_gps or include_position_targets:
        gps_source_column = resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=gps_column,
            fallback_column=gps_fallback_column,
            label="GPS",
        )
        if bs_gps_column not in all_data.columns:
            raise ValueError(f"BS GPS column '{bs_gps_column}' not found in {csv_path}.")
    if include_gps:
        base_columns.extend([gps_source_column, bs_gps_column])
    elif include_position_targets:
        for column in (gps_source_column, bs_gps_column):
            if column not in base_columns:
                base_columns.append(column)
    if include_lidar:
        lidar_source_column = resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=lidar_column,
            fallback_column=lidar_fallback_column,
            label="LiDAR",
        )
        base_columns.append(lidar_source_column)
    if include_mmwave:
        mmwave_source_column = resolve_source_column(
            all_data,
            csv_path=csv_path,
            column=mmwave_column,
            fallback_column=mmwave_fallback_column,
            label="mmWave",
        )
        if mmwave_source_column not in base_columns:
            base_columns.append(mmwave_source_column)
    return SequenceColumnPlan(
        base_columns=base_columns,
        gps_source_column=gps_source_column,
        bs_gps_column=bs_gps_column,
        lidar_source_column=lidar_source_column,
        mmwave_source_column=mmwave_source_column,
    )


def resolve_source_column(
    all_data: pd.DataFrame,
    *,
    csv_path: Path,
    column: str,
    fallback_column: str | None,
    label: str,
) -> str:
    if column in all_data.columns:
        return column
    if fallback_column and fallback_column in all_data.columns:
        return fallback_column
    raise ValueError(
        f"{label} column '{column}' not found in {csv_path}; "
        f"fallback '{fallback_column}' is also unavailable."
    )


def require_columns(all_data: pd.DataFrame, csv_path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in all_data.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}.")


def build_sequence_windows(
    all_data: pd.DataFrame,
    plan: SequenceColumnPlan,
    *,
    in_len: int,
    out_len: int,
    include_gps: bool,
    include_lidar: bool,
    include_mmwave: bool,
    include_position_targets: bool,
) -> pd.DataFrame:
    all_seq_idx = all_data["seq_index"].unique()
    rows = []
    for seq_idx in all_seq_idx:
        seq = all_data.loc[all_data["seq_index"] == seq_idx, plan.base_columns].reset_index(drop=True)
        start = 0
        while start + in_len + out_len <= seq.shape[0]:
            image = seq["unit1_rgb"].iloc[start : start + in_len].tolist()
            radar = seq["unit1_radar"].iloc[start : start + in_len].tolist()
            gps = seq[plan.gps_source_column].iloc[start : start + in_len].tolist() if include_gps else []
            bs_gps = seq[plan.bs_gps_column].iloc[start : start + in_len].tolist() if include_gps else []
            lidar = seq[plan.lidar_source_column].iloc[start : start + in_len].tolist() if include_lidar else []
            mmwave = seq[plan.mmwave_source_column].iloc[start : start + in_len].tolist() if include_mmwave else []
            in_beam = seq["unit1_pwr_60ghz"].iloc[start : start + in_len].tolist()
            out_beam = seq["unit1_pwr_60ghz"].iloc[start + in_len : start + in_len + out_len].tolist()
            future_gps = (
                seq[plan.gps_source_column].iloc[start + in_len : start + in_len + out_len].tolist()
                if include_position_targets
                else []
            )
            future_bs_gps = (
                seq[plan.bs_gps_column].iloc[start + in_len : start + in_len + out_len].tolist()
                if include_position_targets
                else []
            )
            rows.append(
                image
                + radar
                + gps
                + bs_gps
                + lidar
                + mmwave
                + in_beam
                + out_beam
                + future_gps
                + future_bs_gps
                + [seq_idx]
            )
            start += 1
    return pd.DataFrame(
        rows,
        columns=sequence_window_columns(
            in_len,
            out_len,
            include_gps=include_gps,
            include_lidar=include_lidar,
            include_mmwave=include_mmwave,
            include_position_targets=include_position_targets,
        ),
    )


def sequence_window_generation_stats(all_data: pd.DataFrame, *, in_len: int, out_len: int) -> dict:
    raw_sample_count = int(len(all_data))
    generated_window_count = 0
    skipped_history_insufficient = 0
    skipped_future_label_insufficient = 0
    per_seq = {}
    for seq_idx, seq in all_data.groupby("seq_index", sort=False):
        length = int(len(seq))
        generated = max(length - int(in_len) - int(out_len) + 1, 0)
        history_skipped = min(length, max(int(in_len) - 1, 0))
        history_ready_positions = max(length - history_skipped, 0)
        future_skipped = max(history_ready_positions - generated, 0)
        generated_window_count += generated
        skipped_history_insufficient += history_skipped
        skipped_future_label_insufficient += future_skipped
        per_seq[str(seq_idx)] = {
            "raw_sample_count": length,
            "generated_window_count": generated,
            "skipped_history_insufficient": history_skipped,
            "skipped_future_label_insufficient": future_skipped,
            "skipped_cross_sequence": 0,
        }
    return {
        "raw_sample_count": raw_sample_count,
        "generated_window_count": int(generated_window_count),
        "skipped_history_insufficient": int(skipped_history_insufficient),
        "skipped_future_label_insufficient": int(skipped_future_label_insufficient),
        "skipped_cross_sequence": 0,
        "per_seq": per_seq,
    }


def sequence_window_columns(
    in_len: int,
    out_len: int,
    *,
    include_gps: bool,
    include_lidar: bool,
    include_mmwave: bool,
    include_position_targets: bool = False,
) -> list[str]:
    return (
        [f"camera{i}" for i in range(1, in_len + 1)]
        + [f"radar{i}" for i in range(1, in_len + 1)]
        + ([f"gps{i}" for i in range(1, in_len + 1)] if include_gps else [])
        + ([f"bs_gps{i}" for i in range(1, in_len + 1)] if include_gps else [])
        + ([f"lidar{i}" for i in range(1, in_len + 1)] if include_lidar else [])
        + ([f"mmwave{i}" for i in range(1, in_len + 1)] if include_mmwave else [])
        + [f"beam{i}" for i in range(1, in_len + 1)]
        + [f"future_beam{i}" for i in range(1, out_len + 1)]
        + ([f"future_gps{i}" for i in range(1, out_len + 1)] if include_position_targets else [])
        + ([f"future_bs_gps{i}" for i in range(1, out_len + 1)] if include_position_targets else [])
        + ["seq_index"]
    )


def generate_sequence_data(
    csv_path: str | Path,
    data_root: str | Path,
    output_suffix: str,
    in_len: int,
    out_len: int,
    training_set_pct: float = 0.8,
    include_gps: bool = False,
    gps_column: str = "unit2_loc_cal",
    gps_fallback_column: str | None = "unit2_loc",
    bs_gps_column: str = "unit1_loc",
    include_lidar: bool = False,
    lidar_column: str = "unit1_lidar",
    lidar_fallback_column: str | None = "unit1_lidar_SCR",
    include_mmwave: bool = False,
    mmwave_column: str = "unit1_pwr_60ghz",
    mmwave_fallback_column: str | None = None,
    include_position_targets: bool = False,
    split_strategy: str = SPLIT_METADATA_PROTOCOL,
    split_seed: int = 42,
    min_test_sequences: int | None = None,
    test_sequence_count: int | None = None,
    split_metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    all_data = pd.read_csv(csv_path)
    protocol_plan = resolve_split_protocol(split_strategy, in_len=in_len, out_len=out_len)
    if protocol_plan.protocol not in SUPPORTED_SPLIT_METADATA_PROTOCOLS:
        raise ValueError(
            f"Unsupported sequence split_strategy '{split_strategy}'. "
            f"This workflow supports: {sorted(SUPPORTED_SPLIT_METADATA_PROTOCOLS)}."
        )
    plan = resolve_sequence_column_plan(
        all_data,
        csv_path=csv_path,
        include_gps=include_gps,
        gps_column=gps_column,
        gps_fallback_column=gps_fallback_column,
        bs_gps_column=bs_gps_column,
        include_lidar=include_lidar,
        lidar_column=lidar_column,
        lidar_fallback_column=lidar_fallback_column,
        include_mmwave=include_mmwave,
        mmwave_column=mmwave_column,
        mmwave_fallback_column=mmwave_fallback_column,
        include_position_targets=include_position_targets,
    )
    all_seqs = build_sequence_windows(
        all_data,
        plan,
        in_len=in_len,
        out_len=out_len,
        include_gps=include_gps,
        include_lidar=include_lidar,
        include_mmwave=include_mmwave,
        include_position_targets=include_position_targets,
    )
    generation_stats = sequence_window_generation_stats(all_data, in_len=in_len, out_len=out_len)
    split = select_balanced_sequence_split(
        all_seqs,
        training_set_pct=training_set_pct,
        split_seed=split_seed,
        min_test_sequences=min_test_sequences,
        test_sequence_count=test_sequence_count,
        data_root=data_root,
    )
    train_path = data_root / f"train_seqs{output_suffix}.csv"
    test_path = data_root / f"{protocol_plan.eval_file_prefix}_seqs{output_suffix}.csv"
    data_root.mkdir(parents=True, exist_ok=True)
    train_frame = all_seqs[all_seqs["seq_index"].isin(split.train_seq_index)]
    test_frame = all_seqs[all_seqs["seq_index"].isin(split.test_seq_index)]
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    metadata_path = Path(split_metadata_path) if split_metadata_path is not None else default_split_metadata_path(train_path)
    if not metadata_path.is_absolute():
        metadata_path = data_root / metadata_path
    write_split_metadata(
        metadata_path,
        source_csv_path=csv_path,
        data_root=data_root,
        train_path=train_path,
        test_path=test_path,
        all_windows=all_seqs,
        train_frame=train_frame,
        test_frame=test_frame,
        split=split,
        protocol=protocol_plan.protocol,
        eval_name=protocol_plan.eval_name,
        in_len=in_len,
        out_len=out_len,
        enabled_columns=sequence_window_columns(
            in_len,
            out_len,
            include_gps=include_gps,
            include_lidar=include_lidar,
            include_mmwave=include_mmwave,
            include_position_targets=include_position_targets,
        ),
        include_position_targets=include_position_targets,
        window_generation_stats=generation_stats,
        training_set_pct=training_set_pct,
        split_seed=split_seed,
        min_test_sequences=min_test_sequences,
        requested_test_sequence_count=test_sequence_count,
    )
    return train_path, test_path



@PREPROCESSORS.register("sequence_csv")
class SequencePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return generate_sequence_data(**self.kwargs)


__all__ = [
    "SequenceColumnPlan",
    "SequencePreprocessor",
    "SequenceSplit",
    "SplitProtocolPlan",
    "build_sequence_windows",
    "generate_sequence_data",
    "label_distribution_summary",
    "resolve_sequence_column_plan",
    "select_balanced_sequence_split",
    "sequence_window_columns",
    "sequence_window_generation_stats",
    "write_split_metadata",
]
