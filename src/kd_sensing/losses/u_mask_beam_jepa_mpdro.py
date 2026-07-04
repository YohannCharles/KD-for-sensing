import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.training_extensions import ExtensionContext
from kd_sensing.utils.missing_patterns import canonical_missing_pattern_name, list_standard_missing_patterns


def mpdro_enabled(cfg: dict[str, Any]) -> bool:
    raw = cfg.get("mpdro", {})
    return isinstance(raw, dict) and bool(raw.get("enabled", False))


def mpdro_cfg(cfg: dict[str, Any], modalities: tuple[str, ...] | None = None) -> dict[str, Any]:
    raw = cfg.get("mpdro", {})
    raw = raw if isinstance(raw, dict) else {}
    patterns = raw.get("patterns")
    if patterns is None:
        patterns = ["full", "missing_gps", "missing_radar", "radar_only", "lidar_only"]
    if modalities:
        valid = set(core_pattern_names(modalities))
        patterns = [canonical_missing_pattern_name(item) for item in patterns if canonical_missing_pattern_name(item) in valid]
    else:
        patterns = [canonical_missing_pattern_name(item) for item in patterns]
    tau = max(float(raw.get("tau", 1.0)), 1e-6)
    beta = min(max(float(raw.get("ema_beta", 0.9)), 0.0), 0.9999)
    return {
        "patterns": list(dict.fromkeys(patterns)),
        "tau": tau,
        "lambda_dro": min(max(float(raw.get("lambda_dro", 1.0)), 0.0), 1.0),
        "ema_beta": beta,
        "warmup_epochs": max(int(raw.get("warmup_epochs", 3)), 0),
        "detach_weights": bool(raw.get("detach_weights", True)),
        "full_protection": bool(raw.get("full_protection", False)),
        "min_full_weight": min(max(float(raw.get("min_full_weight", 0.10)), 0.0), 1.0),
    }


def new_mpdro_state() -> dict[str, Any]:
    return {
        "ema_loss": {},
        "num_batches": Counter(),
        "last_weights": {},
        "last_raw_weights": {},
        "last_protected_weights": {},
        "last_batch_loss": {},
    }


def mpdro_state(state: dict[str, Any]) -> dict[str, Any]:
    mpdro = state.setdefault("mpdro", new_mpdro_state())
    mpdro.setdefault("ema_loss", {})
    mpdro.setdefault("num_batches", Counter())
    mpdro.setdefault("last_weights", {})
    mpdro.setdefault("last_raw_weights", {})
    mpdro.setdefault("last_protected_weights", {})
    mpdro.setdefault("last_batch_loss", {})
    return mpdro


def mpdro_sample_weights(
    cfg: dict[str, Any],
    state: Any,
    pattern_names: list[str] | None,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    epoch: int,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not mpdro_enabled(cfg) or not isinstance(state, dict) or not pattern_names:
        return None, {}
    canonical_names = [canonical_missing_pattern_name(name) for name in pattern_names]
    cfg_mpdro = mpdro_cfg(cfg)
    patterns = cfg_mpdro["patterns"] or sorted(set(canonical_names))
    if not patterns:
        return None, {}
    mpdro = mpdro_state(state)
    weights = mpdro_group_weights(mpdro, patterns, cfg=cfg_mpdro, epoch=epoch)
    per_sample = per_sample_beam_ce(logits, labels).detach()
    counts = Counter(canonical_names)
    sample_weights = []
    for name in canonical_names:
        sample_weights.append(float(weights.get(name, 0.0)) / max(int(counts.get(name, 0)), 1))
    tensor = torch.tensor(sample_weights, device=logits.device, dtype=logits.dtype)
    if cfg_mpdro["detach_weights"]:
        tensor = tensor.detach()

    beta = float(cfg_mpdro["ema_beta"])
    epoch_batches = state.setdefault("mpdro_epoch_batches", Counter())
    for pattern in sorted(set(canonical_names)):
        mask = torch.tensor([name == pattern for name in canonical_names], dtype=torch.bool, device=logits.device)
        if not bool(mask.any().item()):
            continue
        current = float(per_sample[mask].mean().detach().cpu().item())
        previous = mpdro["ema_loss"].get(pattern)
        mpdro["ema_loss"][pattern] = current if previous is None else beta * float(previous) + (1.0 - beta) * current
        mpdro["num_batches"][pattern] += 1
        epoch_batches[pattern] += 1
        mpdro["last_batch_loss"][pattern] = current
    mpdro["last_weights"] = weights

    diagnostics: dict[str, float] = {}
    for pattern in patterns:
        diagnostics[f"mpdro/weight/{pattern}"] = float(weights.get(pattern, 0.0))
        if pattern in mpdro["ema_loss"]:
            diagnostics[f"mpdro/ema_loss/{pattern}"] = float(mpdro["ema_loss"][pattern])
    return tensor, diagnostics


def mpdro_group_weights(
    mpdro: dict[str, Any],
    patterns: list[str],
    *,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    uniform = uniform_weights(patterns)
    if int(epoch) < int(cfg["warmup_epochs"]):
        raw = uniform
    else:
        ema = mpdro.get("ema_loss", {})
        if any(pattern not in ema for pattern in patterns):
            raw = uniform
        else:
            values = torch.tensor([float(ema[pattern]) for pattern in patterns], dtype=torch.float32)
            weights = torch.softmax(values / float(cfg["tau"]), dim=0)
            raw = {pattern: float(value) for pattern, value in zip(patterns, weights.tolist())}
    protected = mpdro_full_protected_weights(raw, cfg)
    lam = float(cfg.get("lambda_dro", 1.0))
    mixed = {pattern: (1.0 - lam) * uniform.get(pattern, 0.0) + lam * protected.get(pattern, 0.0) for pattern in patterns}
    mixed = renormalize_weights(mixed)
    mpdro["last_raw_weights"] = raw
    mpdro["last_protected_weights"] = protected
    return mixed


def mpdro_full_protected_weights(weights: dict[str, float], cfg: dict[str, Any]) -> dict[str, float]:
    protected = renormalize_weights(dict(weights))
    if not bool(cfg.get("full_protection", False)) or "full" not in protected:
        return protected
    minimum = float(cfg.get("min_full_weight", 0.10))
    if protected.get("full", 0.0) >= minimum:
        return protected
    others = [pattern for pattern in protected if pattern != "full"]
    other_total = sum(protected[pattern] for pattern in others)
    protected["full"] = minimum
    if other_total <= 0.0:
        even = (1.0 - minimum) / max(len(others), 1)
        for pattern in others:
            protected[pattern] = even
        return protected
    scale = (1.0 - minimum) / other_total
    for pattern in others:
        protected[pattern] *= scale
    return renormalize_weights(protected)


def renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {pattern: max(float(value), 0.0) for pattern, value in weights.items()}
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        return uniform_weights(list(clean))
    return {pattern: value / total for pattern, value in clean.items()}


def uniform_weights(patterns: list[str]) -> dict[str, float]:
    if not patterns:
        return {}
    value = 1.0 / len(patterns)
    return {pattern: value for pattern in patterns}


def per_sample_beam_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    labels = labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    labels = labels[:, : logits.shape[1]]
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").view(
        logits.shape[0], -1
    )
    valid = labels.ne(-100)
    return (per_token * valid.to(dtype=per_token.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def write_mpdro_group_log(context: ExtensionContext, state: dict[str, Any], *, epoch: int) -> Path:
    cfg = state.get("config", {})
    modalities = tuple(getattr(context.primary_model, "modalities", context.model_cfg.get("primary", {}).get("modalities", ())))
    cfg_mpdro = mpdro_cfg(cfg, modalities)
    patterns = cfg_mpdro["patterns"]
    mpdro = mpdro_state(state)
    weights = dict(mpdro.get("last_weights") or uniform_weights(patterns))
    raw_weights = dict(mpdro.get("last_raw_weights") or weights)
    protected_weights = dict(mpdro.get("last_protected_weights") or weights)
    counts = state.get("mpdro_epoch_batches")
    if not isinstance(counts, Counter):
        counts = Counter()
    rows = [
        {
            "epoch": int(epoch) + 1,
            "pattern": pattern,
            "ema_loss": csv_float(mpdro["ema_loss"].get(pattern)),
            "raw_weight": csv_float(raw_weights.get(pattern, 0.0)),
            "protected_weight": csv_float(protected_weights.get(pattern, 0.0)),
            "weight": csv_float(weights.get(pattern, 0.0)),
            "num_batches": int(counts.get(pattern, 0)),
        }
        for pattern in patterns
    ]
    path = context.run_dir / "mpdro_mild_group_log.csv"
    append_mpdro_rows(path, rows)
    append_mpdro_rows(context.run_dir / "mpdro_group_log.csv", rows)
    summary = " ".join(f"{pattern}={weights.get(pattern, 0.0):.3f}" for pattern in patterns)
    print(f"[MPDRO] epoch={int(epoch) + 1} weights: {summary}")
    return path


def append_mpdro_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "pattern", "ema_loss", "raw_weight", "protected_weight", "weight", "num_batches"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def csv_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.8g}"


def core_pattern_names(modalities: tuple[str, ...]) -> list[str]:
    standard = list_standard_missing_patterns(modalities, include_avg=False)
    preferred = [
        "full",
        "missing_gps",
        "missing_image",
        "missing_radar",
        "missing_lidar",
        "non_gps_only",
        "gps_only",
        "image_only",
        "radar_only",
        "lidar_only",
    ]
    return [name for name in preferred if name in standard]
