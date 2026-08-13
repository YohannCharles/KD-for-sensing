#!/usr/bin/env python3
"""Benchmark the frozen MMW methods without touching the outer test split.

This is deliberately a narrow, read-only helper.  It builds one validation
batch, loads each published ``validation_best`` checkpoint strictly, and
measures the model forward on that preloaded batch.  The reported FLOP count
is the subset covered by :class:`torch.utils.flop_counter.FlopCounterMode`;
custom or unsupported operators are not silently treated as zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.flop_counter import FlopCounterMode

from kd_sensing.config import load_config
from kd_sensing.engine.batch import prepare_fusion_inputs
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_split_dataset,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_model
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS: tuple[tuple[str, str, str], ...] = (
    (
        "Prototype-only",
        "outputs/four_modal_topology_predictor_masked_feature_fusion/masked_feature_fusion_prototype_only_seed1/resolved_config.yaml",
        "outputs/four_modal_topology_predictor_masked_feature_fusion/masked_feature_fusion_prototype_only_seed1/checkpoints/best.pth",
    ),
    (
        "Hard",
        "outputs/four_modal_topology_predictor_masked_feature_fusion/masked_feature_fusion_off_seed1/resolved_config.yaml",
        "outputs/four_modal_topology_predictor_masked_feature_fusion/masked_feature_fusion_off_seed1/checkpoints/best.pth",
    ),
    (
        "RMBP-MM-local",
        "outputs/mmw_sensing_baselines_no_history_v2/rmbp_mm/train_seed1/resolved_config.yaml",
        "outputs/mmw_sensing_baselines_no_history_v2/rmbp_mm/train_seed1/checkpoints/best.pth",
    ),
    (
        "AMBER-Full-local",
        "outputs/mmw_sensing_baselines_no_history_v2/amber_full/train_seed1/resolved_config.yaml",
        "outputs/mmw_sensing_baselines_no_history_v2/amber_full/train_seed1/checkpoints/best.pth",
    ),
)

RF_TABLE: tuple[dict[str, Any], ...] = (
    {
        "policy": "TBCP-3",
        "beam_measurements": 3,
        "measurement_rounds": 2,
        "feedback_updates": 1,
        "measurement_reduction_vs_full64": (64 - 3) / 64,
        "execution_note": "logical 2+1 metadata; real RF parallelism is hardware-dependent",
        "latency_claim": "beam-count reduction only; not measured wall-clock latency",
    },
    {
        "policy": "Batch-TBCP-2+1",
        "beam_measurements": 3,
        "measurement_rounds": 2,
        "feedback_updates": 1,
        "measurement_reduction_vs_full64": (64 - 3) / 64,
        "execution_note": "two rounds with one controller feedback update",
        "latency_claim": "two measurement rounds/one controller update; physical parallelism is hardware-dependent",
    },
    {
        "policy": "Full-64",
        "beam_measurements": 64,
        "measurement_rounds": 64,
        "feedback_updates": 0,
        "measurement_reduction_vs_full64": 0.0,
        "execution_note": "reference budget",
        "latency_claim": "reference budget",
    },
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    config: str
    checkpoint: str


@dataclass
class BenchmarkRow:
    method: str
    config: str
    config_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_role: str
    seed: int
    parameter_count: int
    trainable_parameter_count: int
    profiler_covered_flops: int | None
    flops_status: str
    flops_error: str | None
    flops_unsupported_note: str
    forward_median_ms: float
    forward_p95_ms: float
    warmup: int
    repeats: int
    batch_size: int
    sequence_length: int
    modalities: tuple[str, ...]
    device: str
    device_name: str
    device_capability: tuple[int, int] | None
    dtype: str
    split: str
    outer_test_accessed: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark frozen MMW validation checkpoints. Only one preloaded validation "
            "batch is used; the outer test split is never built."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        metavar=("NAME", "CONFIG", "CHECKPOINT"),
        help="Override a method (repeatable); defaults to the four frozen seed-1 paths.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu (default: cuda if available).")
    parser.add_argument("--output", default="outputs/mmw_frozen_complexity_benchmark", help="Output directory.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of untimed forward warmups (default: 20).")
    parser.add_argument("--repeats", type=int, default=100, help="Number of timed forwards (default: 100).")
    parser.add_argument("--batch-size", type=int, default=1, help="Validation batch size (default: 1).")
    return parser


def resolve_method_specs(values: Sequence[Sequence[str]] | None) -> list[MethodSpec]:
    raw = values if values else DEFAULT_METHODS
    specs = [MethodSpec(str(name), str(config), str(checkpoint)) for name, config, checkpoint in raw]
    if not specs:
        raise ValueError("At least one frozen method is required.")
    if len({item.name for item in specs}) != len(specs):
        raise ValueError("Method names must be unique in one benchmark run.")
    return specs


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_metadata(device: torch.device) -> tuple[str, tuple[int, int] | None]:
    if device.type != "cuda":
        return ("CPU", None)
    index = int(torch.cuda.current_device() if device.index is None else device.index)
    return (
        str(torch.cuda.get_device_name(index)),
        tuple(int(value) for value in torch.cuda.get_device_capability(index)),
    )


def _checkpoint_metadata(path: Path) -> tuple[dict[str, Any], str]:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    role = str(payload.get("checkpoint_role", ""))
    if role != "validation_best":
        raise ValueError(f"Frozen complexity benchmark requires validation_best, got {role!r}: {path}")
    validate_checkpoint_publication(path, payload=payload)
    return dict(payload), checkpoint_file_digest(path)[0]


def _normalization_metadata(cfg: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any] | None:
    checkpoint_artifacts = payload.get("normalization_artifacts")
    configured = cfg.get("data", {}).get("normalization_artifacts")
    if checkpoint_artifacts and configured and checkpoint_artifacts != configured:
        raise ValueError("Checkpoint and config normalization artifacts do not match.")
    artifacts = checkpoint_artifacts or configured
    return {"normalization_artifacts": artifacts} if artifacts else None


def _build_validation_loader(
    cfg: dict[str, Any],
    *,
    normalization_metadata: dict[str, Any] | None,
    batch_size: int,
):
    validate_normalization_artifact_fingerprint(cfg, normalization_metadata)
    overrides = load_normalization_artifacts(normalization_metadata)
    # A validation-only loader still needs the train-fitted GPS scaler.  MMW's
    # pooled-domain builder expects that scaler on each leaf before indexing.
    dataset = build_split_dataset(cfg, "validation", normalization_overrides=overrides or None)
    loader_cfg = dict(cfg["data"]["dataloader"])
    loader_cfg.update(
        {
            "validation_batch_size": int(batch_size),
            "validation_drop_last": False,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        }
    )
    return build_dataloader(dataset, loader_cfg, split="validation", experiment_seed=None)


def _float32_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point():
            result[key] = value.float()
        else:
            result[key] = value
    return result


def _prepare_model_inputs(raw_batch: Mapping[str, Any], model_cfg: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Prepare complete-modal inputs and remove dataset-side mask metadata.

    A complexity benchmark must measure the full-modality forward, not whatever
    stochastic/temporal mask metadata happens to be attached to a validation
    row.  ``force_modality_mask`` is therefore explicit and all mask keys
    emitted by the DataLoader are intentionally excluded.
    """

    modalities = tuple(str(value) for value in model_cfg.get("modalities", ("image", "radar", "gps", "lidar")))
    if modalities != ("image", "radar", "gps", "lidar"):
        raise ValueError(f"MMW complexity benchmark requires the canonical four modalities, got {modalities}.")
    sequence_length = int(model_cfg.get("seq_length", 5))
    prepared = prepare_fusion_inputs(
        raw_batch,
        seq_length=sequence_length,
        device=device,
        non_blocking=False,
        modalities=modalities,
    )
    input_keys = {"image_batch", "radar_batch", "gps_batch", "lidar_batch"}
    result = {key: value for key, value in prepared.items() if key in input_keys}
    image = result.get("image_batch")
    if not torch.is_tensor(image):
        raise ValueError("Full-modality benchmark requires image_batch.")
    result["force_modality_mask"] = torch.ones(
        int(image.shape[0]),
        len(modalities),
        dtype=torch.bool,
        device=device,
    )
    return _float32_inputs(result)


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _forward(model: torch.nn.Module, inputs: Mapping[str, Any]) -> Any:
    return model(**inputs)


def _measure_forward(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be positive.")
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(warmup):
            _forward(model, inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples: list[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                with torch.cuda.device(device):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _forward(model, inputs)
                    end.record()
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
            else:
                begin = time.perf_counter()
                _forward(model, inputs)
                samples.append((time.perf_counter() - begin) * 1000.0)
    return float(statistics.median(samples)), _percentile(samples, 0.95)


def _profile_flops(model: torch.nn.Module, inputs: Mapping[str, Any]) -> tuple[int | None, str, str | None, str]:
    unsupported_note = (
        "FlopCounterMode counts only registered PyTorch operators; custom or unsupported operators are excluded "
        "and are not interpreted as zero FLOPs."
    )
    try:
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            _forward(model, inputs)
        return int(counter.get_total_flops()), "profiler_covered", None, unsupported_note
    except Exception as exc:  # pragma: no cover - depends on torch/operator versions.
        return None, "failed", f"{type(exc).__name__}: {exc}", unsupported_note


def _model_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Frozen MMW benchmark requires data_protocol provenance.")
    if protocol.get("protocol_id") != "mmw_id_stratified_block_v1":
        raise ValueError("Frozen MMW benchmark requires mmw_id_stratified_block_v1.")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("Frozen benchmark refuses configs whose outer test is already evaluated.")
    return {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "split_manifest_hash": protocol.get("split_manifest_hash"),
        "topology": cfg.get("model", {}).get("primary", {}).get("prototype_topology_id"),
        "outer_test_accessed": False,
    }


def _validate_checkpoint_binding(
    cfg: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    method: str,
) -> None:
    checkpoint_protocol = payload.get("data_protocol")
    config_protocol = cfg.get("data_protocol")
    if not isinstance(checkpoint_protocol, Mapping) or not isinstance(config_protocol, Mapping):
        raise ValueError(f"{method} checkpoint/config is missing data protocol provenance.")
    for key in ("protocol_id", "protocol_fingerprint", "split_manifest_hash", "window_config_hash"):
        if checkpoint_protocol.get(key) != config_protocol.get(key):
            raise ValueError(f"{method} checkpoint protocol field {key!r} does not match its config.")
    cfg_seed = int(cfg.get("experiment", {}).get("seed", -1))
    checkpoint_seed = int(payload.get("experiment_seed", -1))
    if cfg_seed != checkpoint_seed:
        raise ValueError(f"{method} checkpoint seed {checkpoint_seed} does not match config seed {cfg_seed}.")
    if payload.get("checkpoint_role") != "validation_best":
        raise ValueError(f"{method} checkpoint is not a validation_best publication.")


def benchmark_method(spec: MethodSpec, *, device: torch.device, warmup: int, repeats: int, batch_size: int) -> BenchmarkRow:
    config_path = _repo_path(spec.config).resolve()
    checkpoint_path = _repo_path(spec.checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Frozen method config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Frozen method checkpoint not found: {checkpoint_path}")
    cfg = load_config(config_path)
    provenance = _model_provenance(cfg)
    payload, checkpoint_sha256 = _checkpoint_metadata(checkpoint_path)
    _validate_checkpoint_binding(cfg, payload, method=spec.name)
    normalization_metadata = _normalization_metadata(cfg, payload)
    loader = _build_validation_loader(
        cfg,
        normalization_metadata=normalization_metadata,
        batch_size=batch_size,
    )
    try:
        raw_batch = next(iter(loader))
        model = build_model(cfg["model"]["primary"]).to(device).float()
        load_model_state(
            checkpoint_path,
            model,
            role=f"complexity_{spec.name}",
            map_location=device,
            strict=True,
        )
        model.eval()
        model_cfg = cfg["model"]["primary"]
        inputs = _prepare_model_inputs(raw_batch, model_cfg, device)
        sequence_length = int(model_cfg.get("seq_length", cfg.get("data", {}).get("dataset", {}).get("seq_len", 5)))
        with torch.inference_mode():
            output = _forward(model, inputs)
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        if not torch.is_tensor(logits):
            raise ValueError(f"Frozen method {spec.name} did not return logits.")
        profiler_flops, flops_status, flops_error, flops_unsupported_note = _profile_flops(model, inputs)
        median_ms, p95_ms = _measure_forward(
            model,
            inputs,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        params = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        seed = int(cfg.get("experiment", {}).get("seed", payload.get("experiment_seed", -1)))
        device_name, device_capability = _device_metadata(device)
        return BenchmarkRow(
            method=spec.name,
            config=str(config_path),
            config_sha256=_sha256_file(config_path),
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_role=str(payload.get("checkpoint_role")),
            seed=seed,
            parameter_count=params,
            trainable_parameter_count=trainable,
            profiler_covered_flops=profiler_flops,
            flops_status=flops_status,
            flops_error=flops_error,
            flops_unsupported_note=flops_unsupported_note,
            forward_median_ms=median_ms,
            forward_p95_ms=p95_ms,
            warmup=int(warmup),
            repeats=int(repeats),
            batch_size=int(batch_size),
            sequence_length=sequence_length,
            modalities=tuple(str(value) for value in model_cfg.get("modalities", ("image", "radar", "gps", "lidar"))),
            device=str(device),
            device_name=device_name,
            device_capability=device_capability,
            dtype="float32",
            split="validation",
            outer_test_accessed=bool(provenance["outer_test_accessed"]),
        )
    finally:
        shutdown_dataloader_workers(loader)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [_json_ready(item) for item in value]
    return value


def _write_reports(output_dir: Path, rows: Sequence[BenchmarkRow], *, cli: Mapping[str, Any]) -> dict[str, str]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [_json_ready(asdict(row)) for row in rows]
    provenance = {
        "schema_version": 1,
        "dataset": "MMW",
        "split": "validation",
        "outer_test_accessed": False,
        "cli": _json_ready(dict(cli)),
        "defaults": {
            "batch_size": 1,
            "sequence_length": 5,
            "dtype": "float32",
            "device_note": "A40 is the intended reference device; actual device is recorded per row",
            "warmup": 20,
            "repeats": 100,
        },
        "methods": row_dicts,
        "rf_measurement_budget": list(RF_TABLE),
        "latency_scope": "Forward timing excludes RF measurement/controller latency; only beam-count and round/update accounting is reported.",
        "runtime_measurement_note": "Median/p95 are repeated-forward wall-clock measurements on the recorded device; runtime variance is expected even with CUDA-event synchronization.",
        "flops_scope": "profiler-covered FLOPs from one torch.utils.flop_counter.FlopCounterMode forward; unsupported/custom operators are excluded.",
    }
    json_path = output_dir / "benchmark.json"
    json_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "benchmark.csv"
    fields = list(row_dicts[0]) if row_dicts else [field.name for field in BenchmarkRow.__dataclass_fields__.values()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in row_dicts:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    md_path = output_dir / "benchmark.md"
    lines = [
        "# MMW frozen-method complexity benchmark",
        "",
        "仅使用预加载 validation batch；`outer_test_accessed=false`。FLOPs 是 profiler-covered 下界，未覆盖的自定义/不支持算子不计入。",
        "",
        "| Method | Params | Profiler FLOPs | Median (ms) | P95 (ms) | Device | Config SHA | Checkpoint |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        flops = "n/a" if row.profiler_covered_flops is None else f"{row.profiler_covered_flops:,}"
        lines.append(
            f"| {row.method} | {row.parameter_count:,} | {flops} | {row.forward_median_ms:.4f} | {row.forward_p95_ms:.4f} | {row.device_name} ({row.device}) | `{row.config_sha256[:12]}` | `{row.checkpoint_sha256[:12]}` |"
        )
    lines += [
        "",
        "## RF probing budget (not wall-clock latency)",
        "",
        "| Policy | Beam measurements | Measurement rounds | Controller feedback updates | Reduction vs Full-64 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in RF_TABLE:
        reduction = f"{100.0 * float(item['measurement_reduction_vs_full64']):.4f}%"
        lines.append(
            f"| {item['policy']} | {item['beam_measurements']} | {item['measurement_rounds']} | {item['feedback_updates']} | {reduction} |"
        )
    lines += [
        "",
        "Batch-TBCP-2+1 的物理含义是 2 个 measurement rounds、1 次 controller feedback update；是否减少真实 RF 时隙取决于硬件能否并行/可分离测量。",
        "",
        "前向计时只反映已加载模型的计算路径；同一架构不同运行的时间差异属于 runtime variance，应结合 median/p95 解读。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0 or args.batch_size <= 0:
        raise SystemExit("--warmup must be >= 0, --repeats and --batch-size must be positive.")
    specs = resolve_method_specs(args.method)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested CUDA device {device}, but CUDA is unavailable.")
    rows = []
    for spec in specs:
        rows.append(
            benchmark_method(
                spec,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                batch_size=args.batch_size,
            )
        )
    paths = _write_reports(Path(args.output).resolve(), rows, cli=vars(args))
    print(json.dumps({"output": paths, "methods": [row.method for row in rows], "outer_test_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
