import platform
import time
from typing import Any, Mapping

import torch


PAPER_TARGET_DBA = {
    "S32": 0.8660,
    "S33": 0.8627,
    "S34": 0.8670,
    "overall": 0.8652,
}


def build_bev_fusion_2604_report(
    scene_metrics: Mapping[str, Mapping[str, Any]],
    *,
    split_protocol: str | None = None,
    seed: int | None = None,
    metric_profile: str = "2604_linear_topk",
    paper_exact_split_available: bool = False,
    mock_data: bool = False,
    paper_approximation: bool = False,
    hardware: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable arXiv:2604.05668 BEV-Fusion report payload."""

    rows: dict[str, dict[str, Any]] = {}
    weighted_sum = 0.0
    weighted_count = 0
    macro_values: list[float] = []
    for raw_scene, metrics in scene_metrics.items():
        scene = _scene_key(raw_scene)
        linear_dba = _optional_float(
            metrics.get("linear_dba", metrics.get("DBA", metrics.get("dba", metrics.get("test_dba"))))
        )
        sample_count = int(metrics.get("sample_count", metrics.get("valid_label_count", metrics.get("num_samples", 0))) or 0)
        topk = _topk_payload(metrics)
        target = PAPER_TARGET_DBA.get(scene)
        rows[scene] = {
            "linear_dba": linear_dba,
            "topk": topk,
            "sample_count": sample_count,
            "paper_target_linear_dba": target,
            "gap_to_paper": None if linear_dba is None or target is None else float(linear_dba - target),
            "metric_profile": metric_profile,
        }
        if linear_dba is not None:
            macro_values.append(float(linear_dba))
            if sample_count > 0:
                weighted_sum += float(linear_dba) * sample_count
                weighted_count += sample_count
    macro = sum(macro_values) / len(macro_values) if macro_values else None
    weighted = weighted_sum / weighted_count if weighted_count > 0 else macro
    report = {
        "report_type": "bev_fusion_2604_reproduction",
        "metric_profile": metric_profile,
        "dba_distance_mode": "linear",
        "scene_breakdown": rows,
        "macro_linear_dba": macro,
        "weighted_overall_linear_dba": weighted,
        "paper_target": dict(PAPER_TARGET_DBA),
        "gap_to_paper_overall": None if weighted is None else float(weighted - PAPER_TARGET_DBA["overall"]),
        "split_protocol": split_protocol,
        "seed": seed,
        "paper_exact_split_available": bool(paper_exact_split_available),
        "mock_data": bool(mock_data),
        "paper_approximation": bool(paper_approximation),
        "hardware": dict(hardware or {}),
        "model": dict(model_metadata or {}),
    }
    if mock_data:
        report["caveat"] = "mock_or_synthetic_metrics_must_not_be_reported_as_real_deepsense6g_reproduction"
    elif not paper_exact_split_available:
        report["caveat"] = "paper_exact_split_seed_code_and_weights_are_not_available"
    return report


def bev_fusion_2604_model_size(model: torch.nn.Module) -> dict[str, float | int]:
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_m": total / 1_000_000.0,
        "trainable_params_m": trainable / 1_000_000.0,
    }


def local_hardware_summary() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(index))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_devices": gpus,
        "h100_claim": False,
    }


def measure_local_forward_latency_ms(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    runs: int = 20,
    warmup: int = 5,
) -> dict[str, Any]:
    """Optional local latency probe; it is not a paper H100 latency claim."""

    model.eval()
    device = next(model.parameters()).device
    prepared = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        for _ in range(max(int(warmup), 0)):
            model(**prepared)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(max(int(runs), 1)):
            start = time.perf_counter()
            model(**prepared)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - start) * 1000.0)
    values = torch.tensor(durations, dtype=torch.float64)
    return {
        "latency_profile": "local_forward_wall_clock",
        "runs": int(len(durations)),
        "warmup": int(max(warmup, 0)),
        "mean_ms": float(values.mean().item()),
        "median_ms": float(values.median().item()),
        "hardware": local_hardware_summary(),
        "paper_h100_latency_claim": False,
    }


def _scene_key(raw: str) -> str:
    text = str(raw).strip().upper()
    if text.startswith("SCENE"):
        text = "S" + text.replace("SCENE", "", 1)
    if text.isdigit():
        text = f"S{text}"
    return text


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _topk_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    topk = metrics.get("topk")
    payload: dict[str, float] = {}
    if isinstance(topk, Mapping):
        for key, value in topk.items():
            payload[str(key)] = float(value)
    for key in ("top1", "top3", "top5", "top10"):
        if key in metrics:
            payload[key.replace("top", "")] = float(metrics[key])
    return payload


__all__ = [
    "PAPER_TARGET_DBA",
    "bev_fusion_2604_model_size",
    "build_bev_fusion_2604_report",
    "local_hardware_summary",
    "measure_local_forward_latency_ms",
]
