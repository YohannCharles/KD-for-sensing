#!/usr/bin/env python
"""Decide whether mask-conditional conformal calibration is needed at all.

Read-only.  Nothing trains, no checkpoint is written, the outer test is never
touched.  The frozen U0 cache built for the router observability screen is
replayed under each of the 15 canonical masks, and split-conformal calibration
is run twice: once marginally (a single threshold for everything, which is what
any off-the-shelf application would do) and once per mask.

If the marginal threshold holds the nominal level on every mask, conditioning
on the availability pattern is unnecessary and the set-valued route should be
narrowed accordingly.  If it under-covers on the degraded masks, that gap is
the motivation, measured rather than asserted.

    conda run -n kd_mm_beam python tools/run_conformal_beam_diagnostic.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.baselines import conformal_beam_sets as cbs
from kd_sensing.baselines import router_observability as ro
from kd_sensing.baselines.full_pool_common import atomic_csv, write_json
from kd_sensing.baselines.prototype_decision_adapter import load_frozen_u0, preflight


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/conformal_beam_diagnostic"
CACHE_ROOT = ROOT / "outputs/router_observability/cache"
U0_ROOT = ROOT / "outputs/full_pool_capacity/u0_seed1"
U0_CONFIG = U0_ROOT / "final_config.yaml"
U0_CHECKPOINT = U0_ROOT / "checkpoints/last.pth"
U0_SHA256 = "ed909406a37ec4ccd2b08bd1fb65ab66fc437cec226a526fdaf7ada1407ba8cf"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def mask_probabilities(
    head: ro.FrozenU0Head,
    cache: ro.RepresentationCache,
    pattern: Sequence[int],
) -> np.ndarray:
    """Replay U0's own router under one mask and return softmax probabilities."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(cache), ro.BATCH_SIZE):
        index = torch.arange(start, min(start + ro.BATCH_SIZE, len(cache)), device=cache.device)
        batch = cache.slice(index)
        replayed = head(
            batch["latent_sequence"],
            batch["preprojection"],
            cache.available(index, pattern),
        )
        logits = head.reference_logits(replayed)
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


# Which covariates each scheme conditions on, on top of the mask.  ``marginal``
# conditions on nothing at all -- one threshold for every mask -- and is what an
# off-the-shelf application of split conformal would do.
SCHEMES: tuple[str, ...] = ("marginal", "mask", "mask_weather", "mask_domain")


def diagnose(
    probabilities: Mapping[str, np.ndarray],
    labels: np.ndarray,
    calibration: np.ndarray,
    covariates: Mapping[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every conditioning scheme on the same replay, same split, same alpha."""
    test = ~calibration
    scores = {key: cbs.nonconformity(value) for key, value in probabilities.items()}

    # One threshold pooled over every mask: the assumption under test is that a
    # single number can serve all availability patterns.
    marginal_q = cbs.conformal_quantile(
        np.concatenate(
            [cbs.true_beam_scores(scores[key][calibration], labels[calibration]) for key in scores]
        ),
        alpha,
    )

    rows: list[dict[str, Any]] = []
    for mask, score in scores.items():
        truth = cbs.true_beam_scores(score, labels)
        mask_q = cbs.conformal_quantile(truth[calibration], alpha)
        for scheme in SCHEMES:
            if scheme == "marginal":
                threshold: float | np.ndarray = marginal_q
                unseen = np.zeros(len(labels), dtype=bool)
            elif scheme == "mask":
                threshold, unseen = mask_q, np.zeros(len(labels), dtype=bool)
            else:
                # Finer strata fall back to this mask's own threshold, never to
                # infinity, so an unseen scene costs coverage rather than hiding.
                threshold, unseen = cbs.stratum_thresholds(
                    truth, covariates[scheme], calibration, alpha=alpha, fallback=mask_q
                )
            rows.append(
                {
                    "mask": mask,
                    "scheme": scheme,
                    "test_samples": int(test.sum()),
                    "coverage": cbs.coverage(score[test], labels[test], _restrict(threshold, test)),
                    "mean_set_size": float(
                        np.mean(cbs.set_sizes(score[test], _restrict(threshold, test)))
                    ),
                    "fallback_fraction": float(np.mean(unseen[test])),
                }
            )

    # A fixed top-K sized to the marginal set is the zero-machinery control: the
    # same average sweep budget with no calibration at all.
    marginal_rows = [row for row in rows if row["scheme"] == "marginal"]
    budget = max(1, int(round(float(np.mean([row["mean_set_size"] for row in marginal_rows])))))
    for mask, score in scores.items():
        ranked = np.argsort(-probabilities[mask][test], axis=1)[:, :budget]
        rows.append(
            {
                "mask": mask,
                "scheme": f"fixed_top{budget}",
                "test_samples": int(test.sum()),
                "coverage": float(np.mean((ranked == labels[test][:, None]).any(axis=1))),
                "mean_set_size": float(budget),
                "fallback_fraction": 0.0,
            }
        )

    summary: dict[str, Any] = {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "calibration_samples": int(calibration.sum()),
        "test_samples": int(test.sum()),
        "marginal_threshold": marginal_q,
        "schemes": {},
    }
    for scheme in sorted({row["scheme"] for row in rows}):
        group = [row for row in rows if row["scheme"] == scheme]
        values = [row["coverage"] for row in group]
        summary["schemes"][scheme] = {
            "coverage_min": min(values),
            "coverage_max": max(values),
            "coverage_spread": max(values) - min(values),
            "worst_mask": group[int(np.argmin(values))]["mask"],
            "masks_below_nominal": sum(1 for value in values if value < 1.0 - alpha),
            "mean_set_size": float(np.mean([row["mean_set_size"] for row in group])),
            "fallback_fraction": float(np.mean([row["fallback_fraction"] for row in group])),
        }
    return rows, summary


def _restrict(threshold: float | np.ndarray, selector: np.ndarray) -> float | np.ndarray:
    return threshold if np.ndim(threshold) == 0 else threshold[selector]


SPLIT_NOTES = {
    "track": "标定/测试按 `(domain, cav)` 轨迹整块切分（无泄漏，但两侧看到的是不同场景）。",
    "random": "**对照**：帧级随机切分，故意允许同一轨迹的相邻帧落在两侧（有泄漏，但恢复可交换性）。",
}
SCHEME_NOTES = {
    "marginal": "单一阈值，不做任何条件化",
    "mask": "按 15 种可用性组合分层",
    "mask_weather": "按 mask × 天气分层（45 层）",
    "mask_domain": "按 mask × 场景分层（225 层）",
}


def report(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], setting: str, split: str
) -> str:
    nominal = summary["nominal_coverage"]
    masks = [row["mask"] for row in rows if row["scheme"] == "marginal"]
    schemes = list(SCHEMES) + sorted({row["scheme"] for row in rows} - set(SCHEMES))
    lines = [
        "# Conformal 波束集合诊断（条件化方案对比）",
        "",
        "> 冻结 U0，**零训练**：分数取自 U0 自己的 router（`reference_logits`）。",
        f"> 设定 {setting}，开发集，未访问 outer test。{SPLIT_NOTES[split]}",
        f"> 名义覆盖率 {nominal:.2f}（alpha={summary['alpha']}），"
        f"标定 {summary['calibration_samples']} 样本 / 测试 {summary['test_samples']} 样本。",
        "",
        "分数为 `1 - softmax(fused_logits)`。U0 的 `_head_logits` 是对 `BeamPrototypeBank` 的余弦 logits，",
        "因此该分数本身就是原型度量空间里的量，不是外挂的置信度。",
        "",
        "## 方案汇总",
        "",
        "| 方案 | 条件变量 | 覆盖率区间 | 跨 mask 跨度 | 低于名义 | 平均集合 | 回退比例 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for scheme in schemes:
        stat = summary["schemes"][scheme]
        note = SCHEME_NOTES.get(scheme, "等预算零机制对照")
        lines.append(
            f"| `{scheme}` | {note} | [{stat['coverage_min']:.4f}, {stat['coverage_max']:.4f}] "
            f"| {stat['coverage_spread']:.4f} | {stat['masks_below_nominal']}/{len(masks)} "
            f"| {stat['mean_set_size']:.1f} | {stat['fallback_fraction']:.3f} |"
        )
    lines += [
        "",
        "「回退比例」是测试样本中所属分层在标定集里不存在、只能退回 mask 级阈值的占比。",
        "",
        "## 逐 mask 覆盖率",
        "",
        "| mask | " + " | ".join(f"`{scheme}`" for scheme in schemes) + " |",
        "|---" * (len(schemes) + 1) + "|",
    ]
    lookup = {(row["mask"], row["scheme"]): row for row in rows}
    for mask in masks:
        cells = []
        for scheme in schemes:
            value = lookup[(mask, scheme)]["coverage"]
            cells.append(f"{value:.4f}" + ("" if value >= nominal else " ⚠"))
        lines.append(f"| {mask} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 逐 mask 平均集合大小",
        "",
        "| mask | " + " | ".join(f"`{scheme}`" for scheme in schemes) + " |",
        "|---" * (len(schemes) + 1) + "|",
    ]
    for mask in masks:
        cells = [f"{lookup[(mask, scheme)]['mean_set_size']:.1f}" for scheme in schemes]
        lines.append(f"| {mask} | " + " | ".join(cells) + " |")
    lines += ["", f"生成时间：{now()}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="N", choices=sorted(ro.SETTINGS))
    parser.add_argument("--alpha", type=float, default=cbs.ALPHA)
    parser.add_argument(
        "--split",
        default="track",
        choices=("track", "random"),
        help="track: whole trajectories per side (leak-free). random: frame-level control.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, audit = preflight(U0_CONFIG, U0_CHECKPOINT, U0_SHA256, migration_root=args.output / "protocol")
    model, structure = load_frozen_u0(cfg, U0_CHECKPOINT, device)
    head = ro.FrozenU0Head(model)
    cache = ro.load_cache(CACHE_ROOT / f"setting_{args.setting}" / "validation.npz", device)

    labels = cache.label.cpu().numpy()
    if args.split == "track":
        calibration = cbs.block_split(cache.sample_id.tolist(), cache.domain.tolist())
    else:
        calibration = cbs.random_split(len(cache))
    covariates = {
        "mask_weather": np.asarray(cache.weather, dtype=str),
        "mask_domain": np.asarray(cache.domain, dtype=str),
    }
    probabilities = {
        key: mask_probabilities(head, cache, pattern) for key, pattern in ro.MASK_PATTERNS
    }
    rows, summary = diagnose(probabilities, labels, calibration, covariates, alpha=args.alpha)

    output = args.output / f"split_{args.split}_alpha{args.alpha:g}"
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "per_mask_coverage.csv", rows, list(rows[0]))
    write_json(
        output / "summary.json",
        {
            "setting": args.setting,
            "u0_checkpoint": str(U0_CHECKPOINT),
            "u0_sha256": U0_SHA256,
            "preflight": audit,
            "u0_structure": structure,
            "trained": False,
            "outer_test_accessed": False,
            "split": {
                "mode": args.split,
                "unit": "(domain, cav) trajectory block" if args.split == "track" else "frame",
                "seed": cbs.SPLIT_SEED,
                "calibration_fraction": cbs.CALIBRATION_FRACTION,
                "tracks": int(np.unique(cbs.track_ids(cache.sample_id.tolist(), cache.domain.tolist())).size),
            },
            "summary": summary,
            "created_at": now(),
        },
    )
    text = report(rows, summary, args.setting, args.split)
    (output / "conformal_beam_diagnostic_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
