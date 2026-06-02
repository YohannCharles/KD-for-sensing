from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HISTOGRAM_FIELDS = {
    "absolute": "beam_histogram",
    "geometry": "beam_geo_histogram",
    "residual": "beam_residual_histogram",
}


def analyze_distribution_shift(
    *,
    split_artifact: Mapping[str, Any],
    output_dir: str | Path,
    split_artifact_path: str | Path | None = None,
    label_space: Mapping[str, Any] | None = None,
    smoothing: float = 1e-6,
    make_figures: bool = False,
    figures_required: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_cfg = dict(label_space or {})
    stats = split_artifact.get("stats", {}) if isinstance(split_artifact.get("stats"), Mapping) else {}
    histograms = _collect_histograms(stats)
    metrics = _collect_metrics(histograms, smoothing=float(smoothing), label_space=label_cfg)
    figure_status = _maybe_write_figures(out, histograms, enabled=make_figures, required=figures_required)
    summary = _summary(metrics, label_cfg)
    payload = {
        "version": "beam_distribution_shift_diagnostics_v1",
        "split_artifact_path": str(split_artifact_path or ""),
        "input_fingerprint": split_artifact.get("input_fingerprint"),
        "label_space": label_cfg,
        "smoothing": float(smoothing),
        "sample_counts": {
            split: int(payload.get("count", 0))
            for split, payload in split_artifact.get("splits", {}).items()
            if isinstance(payload, Mapping)
        },
        "histograms": histograms,
        "metrics": metrics,
        "summary": summary,
        "figure_generation": figure_status,
        "offline_diagnostics_scope": True,
        "target_unlabeled_label_read_scope": "offline_diagnostics",
    }
    metrics_path = out / "distribution_shift_metrics.json"
    hist_json_path = out / "distribution_shift_histograms.json"
    hist_csv_path = out / "distribution_shift_histograms.csv"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hist_json_path.write_text(json.dumps(histograms, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_histogram_csv(hist_csv_path, histograms)
    summary_path = out / "distribution_shift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["outputs"] = {
        "metrics_json": str(metrics_path),
        "histograms_json": str(hist_json_path),
        "histograms_csv": str(hist_csv_path),
        "summary_json": str(summary_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def distribution_distances(
    source_hist: Mapping[str, int],
    target_hist: Mapping[str, int],
    *,
    smoothing: float = 1e-6,
    circular: bool = False,
) -> dict[str, float]:
    bins = _ordered_bins(source_hist, target_hist)
    p = _probabilities(source_hist, bins, smoothing=smoothing)
    q = _probabilities(target_hist, bins, smoothing=smoothing)
    return {
        "kl": float(np.sum(p * np.log(p / q))),
        "js": float(_js_divergence(p, q)),
        "wasserstein_emd": float(_wasserstein_1d(p, q, circular=circular)),
        "total_variation": float(0.5 * np.sum(np.abs(p - q))),
    }


def _collect_histograms(stats: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, payload in stats.items():
        if not isinstance(payload, Mapping):
            continue
        result[split] = {"count": int(payload.get("count", 0))}
        for label_name, field in HISTOGRAM_FIELDS.items():
            hist = payload.get(field)
            if isinstance(hist, Mapping) and hist:
                result[split][label_name] = {str(key): int(value) for key, value in hist.items()}
    return result


def _collect_metrics(histograms: Mapping[str, Any], *, smoothing: float, label_space: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    source = histograms.get("source", {}) if isinstance(histograms.get("source"), Mapping) else {}
    for target_split in ("target_labeled", "target_unlabeled", "target_test"):
        target = histograms.get(target_split, {}) if isinstance(histograms.get(target_split), Mapping) else {}
        split_metrics: dict[str, Any] = {}
        for label_name in HISTOGRAM_FIELDS:
            source_hist = source.get(label_name)
            target_hist = target.get(label_name)
            if not isinstance(source_hist, Mapping) or not isinstance(target_hist, Mapping):
                continue
            circular = label_name in {"absolute", "geometry"} or str(label_space.get("residual_convention", "")) == "full_circular"
            split_metrics[label_name] = distribution_distances(
                source_hist,
                target_hist,
                smoothing=smoothing,
                circular=circular,
            )
        if split_metrics:
            split_metrics["emd_absolute"] = split_metrics.get("absolute", {}).get("wasserstein_emd")
            split_metrics["emd_residual"] = split_metrics.get("residual", {}).get("wasserstein_emd")
            metrics[target_split] = split_metrics
    metrics["class_order"] = {
        "absolute": "ordered circular beam class",
        "geometry": "ordered circular beam class",
        "residual": f"residual convention={label_space.get('residual_convention', 'signed_circular')}",
    }
    return metrics


def _summary(metrics: Mapping[str, Any], label_space: Mapping[str, Any]) -> dict[str, Any]:
    target = metrics.get("target_test", {}) if isinstance(metrics.get("target_test"), Mapping) else {}
    absolute = target.get("emd_absolute")
    residual = target.get("emd_residual")
    statements = []
    if absolute is not None:
        statements.append(f"emd_absolute={float(absolute):.6g} measures source-to-target_test absolute beam shift.")
    if residual is not None:
        statements.append(
            f"emd_residual={float(residual):.6g} measures source-to-target_test residual beam shift "
            f"under {label_space.get('residual_convention', 'signed_circular')} ordering."
        )
    if absolute is not None and residual is not None and float(residual) < float(absolute):
        statements.append(
            "Residual label space has a smaller label distribution distance for this split; "
            "this diagnostic fact alone is not sufficient evidence of model generalization improvement."
        )
    return {
        "emd_absolute": absolute,
        "emd_residual": residual,
        "statements": statements,
        "claim_boundary": "Distribution distances describe labels only and do not assert model performance gains.",
    }


def _write_histogram_csv(path: Path, histograms: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label_space", "bin", "count"])
        writer.writeheader()
        for split, payload in sorted(histograms.items()):
            if not isinstance(payload, Mapping):
                continue
            for label_name in HISTOGRAM_FIELDS:
                hist = payload.get(label_name)
                if not isinstance(hist, Mapping):
                    continue
                for bin_id, count in sorted(hist.items(), key=lambda item: _numeric_sort_key(item[0])):
                    writer.writerow(
                        {
                            "split": split,
                            "label_space": label_name,
                            "bin": bin_id,
                            "count": int(count),
                        }
                    )


def _maybe_write_figures(
    output_dir: Path,
    histograms: Mapping[str, Any],
    *,
    enabled: bool,
    required: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "skipped_reason": "figure_generation_disabled"}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        if required:
            raise
        return {"enabled": True, "generated": False, "skipped_reason": f"matplotlib_unavailable:{exc}"}
    generated = []
    for label_name in HISTOGRAM_FIELDS:
        fig, ax = plt.subplots(figsize=(8, 3))
        for split, payload in sorted(histograms.items()):
            hist = payload.get(label_name) if isinstance(payload, Mapping) else None
            if not isinstance(hist, Mapping) or not hist:
                continue
            bins = [key for key, _ in sorted(hist.items(), key=lambda item: _numeric_sort_key(item[0]))]
            counts = [int(hist[key]) for key in bins]
            ax.plot(range(len(bins)), counts, marker="o", label=split)
        ax.set_title(f"{label_name} histogram")
        ax.set_xlabel("class")
        ax.set_ylabel("count")
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_dir / f"{label_name}_histogram.png"
        fig.savefig(path)
        plt.close(fig)
        generated.append(str(path))
    return {"enabled": True, "generated": bool(generated), "paths": generated}


def _probabilities(hist: Mapping[str, int], bins: list[str], *, smoothing: float) -> np.ndarray:
    values = np.asarray([float(hist.get(bin_id, 0)) + float(smoothing) for bin_id in bins], dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.ones((len(bins),), dtype=np.float64) / max(len(bins), 1)
    return values / total


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m))) + 0.5 * float(np.sum(q * np.log(q / m)))


def _wasserstein_1d(p: np.ndarray, q: np.ndarray, *, circular: bool) -> float:
    if p.size == 0:
        return 0.0
    if not circular or p.size <= 2:
        return float(np.sum(np.abs(np.cumsum(p - q))))
    diffs = []
    delta = p - q
    for shift in range(p.size):
        diffs.append(float(np.sum(np.abs(np.cumsum(np.roll(delta, shift))))))
    return float(min(diffs))


def _ordered_bins(left: Mapping[str, int], right: Mapping[str, int]) -> list[str]:
    return [key for key in sorted(set(left) | set(right), key=_numeric_sort_key)]


def _numeric_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(float(str(value))))
    except ValueError:
        return (1, str(value))


__all__ = [
    "HISTOGRAM_FIELDS",
    "analyze_distribution_shift",
    "distribution_distances",
]
