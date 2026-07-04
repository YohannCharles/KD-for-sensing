import math
from typing import Any, Mapping

import numpy as np

from kd_sensing.data.mmw.support_selection import angle_coverage_indices


def select_support_samples(
    samples: list[Any],
    adapt_cfg: Mapping[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    mode = str(adapt_cfg.get("support_mode", "temporal_first"))
    count = support_count(
        samples,
        support_ratio=adapt_cfg.get("support_ratio"),
        support_num=adapt_cfg.get("support_num"),
    )
    if count <= 0:
        return [], list(samples), {"selection_mode": mode, "support_count": 0, "query_count": len(samples)}
    if mode == "random":
        rng = np.random.default_rng(int(adapt_cfg.get("seed", 42)))
        indices = sorted(rng.choice(len(samples), size=count, replace=False).tolist())
        support_set = set(indices)
    elif mode == "trajectory":
        ordered_groups: dict[str, list[int]] = {}
        for idx, sample in enumerate(samples):
            key = _sample_attr(sample, "branch_key") or _sample_metadata(sample).get("contiguous_segment_id") or _sample_attr(
                sample, "sample_id"
            )
            ordered_groups.setdefault(str(key), []).append(idx)
        selected: list[int] = []
        for indices in ordered_groups.values():
            selected.extend(indices)
            if len(selected) >= count:
                break
        support_set = set(sorted(selected)[:count])
    elif mode == "temporal_first":
        ordered = sorted(
            range(len(samples)),
            key=lambda idx: (_sample_attr(samples[idx], "order_key", 0.0), _sample_attr(samples[idx], "sample_id", "")),
        )
        support_set = set(ordered[:count])
    elif mode == "angle_coverage":
        support_set = angle_coverage_indices(
            samples,
            count,
            angle_getter=theta_angle_degrees,
            include_extrema=True,
        )
    else:
        raise ValueError("adapt.support_mode must be one of temporal_first, angle_coverage, random, or trajectory.")
    support = [sample for idx, sample in enumerate(samples) if idx in support_set]
    query = [sample for idx, sample in enumerate(samples) if idx not in support_set]
    return support, query, {
        "selection_mode": mode,
        "seed": int(adapt_cfg.get("seed", 42)),
        "support_count": len(support),
        "query_count": len(query),
        "support_ratio": adapt_cfg.get("support_ratio"),
        "support_num": adapt_cfg.get("support_num"),
        "support_num_overrides_ratio": adapt_cfg.get("support_num") not in {None, ""},
        "support_angle_range_degrees": theta_range(support),
        "query_angle_range_degrees": theta_range(query),
    }


def theta_angle_degrees(sample: Any) -> float | None:
    value = float(_sample_attr(sample, "theta_degrees", float("nan")))
    return value if math.isfinite(value) else None


def theta_range(samples: list[Any]) -> list[float] | None:
    values = [float(_sample_attr(sample, "theta_degrees")) for sample in samples if theta_angle_degrees(sample) is not None]
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def support_count(samples: list[Any], *, support_ratio: Any, support_num: Any) -> int:
    total = len(samples)
    explicit = _optional_int(support_num)
    if explicit is not None:
        return max(0, min(explicit, total))
    ratio = 0.0 if support_ratio in {None, ""} else float(support_ratio)
    return max(0, min(int(math.ceil(total * ratio)), total))


def _sample_metadata(sample: Any) -> Mapping[str, Any]:
    value = _sample_attr(sample, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _sample_attr(sample: Any, name: str, default: Any = "") -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


__all__ = [
    "select_support_samples",
    "support_count",
    "theta_angle_degrees",
    "theta_range",
]
