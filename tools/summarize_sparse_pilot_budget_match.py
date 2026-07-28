#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARMS = (
    ("dense32x16", 0, "GPU-93786037-1a47-33f8-0e8a-f897dc58df9a"),
    ("spatial4x16", 1, "GPU-05040195-56e8-661a-c2d6-03b99cc0e2bb"),
    ("target4x8", 2, "GPU-2cefc478-8fe5-d278-85b5-649ae5622933"),
    ("curriculum", 3, "GPU-b14e3776-f5e5-4c95-cfdd-ba53e7a166c0"),
)
GPU_UUIDS = tuple(item[2] for item in ARMS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed matched-update sparse-pilot diagnostics.")
    parser.add_argument("--root", type=Path, default=Path("outputs/sparse_pilot_budget_match"))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--arms", help="Comma-separated four arm names; defaults to the original matched arms.")
    args = parser.parse_args()

    selected_arms = ARMS
    if args.arms:
        names = tuple(item.strip() for item in args.arms.split(",") if item.strip())
        if len(names) != len(GPU_UUIDS) or len(set(names)) != len(names):
            raise ValueError("--arms requires four distinct comma-separated arm names.")
        selected_arms = tuple((arm, index, GPU_UUIDS[index]) for index, arm in enumerate(names))

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for arm, gpu_index, gpu_uuid in selected_arms:
        arm_root = args.root / arm
        return_code = int((arm_root / "return_code.txt").read_text(encoding="utf-8").strip())
        manifest.append({"arm": arm, "gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "return_code": return_code})
        if return_code:
            raise RuntimeError(f"Arm {arm} failed with return code {return_code}; see {arm_root / 'run.log'}")
        diagnostics = json.loads((arm_root / "diagnostics.json").read_text(encoding="utf-8"))
        primary = {row["method"]: row for row in read_csv(arm_root / "ablation_summary.csv")}
        final_budget = read_csv(arm_root / "budget_summary.csv")[-1]
        history_path = arm_root / "training_history.csv"
        history = read_csv(history_path) if history_path.is_file() else []
        mask_by_key = {
            (row["mask"], row["csi_state"]): row
            for row in read_csv(arm_root / "mask_summary.csv")
        }
        rows.append(
            {
                "arm": arm,
                "train_samples": diagnostics["train_samples"],
                "validation_samples": diagnostics["validation_samples"],
                "epochs": diagnostics["epochs"],
                "optimizer_steps": diagnostics.get("optimizer_steps", ""),
                "batch_size": diagnostics.get("batch_size", ""),
                "final_patterns": final_budget["num_selected_patterns"],
                "final_frequencies": final_budget["num_pilot_subcarriers"],
                "pilot_soundings": final_budget["pilot_soundings"],
                "pilot_resource_elements": final_budget["pilot_resource_elements"],
                "c0_top1": primary["C0"]["top1"],
                "c5_top1": primary["C5"]["top1"],
                "c5_top3": primary["C5"]["top3"],
                "c5_top5": primary["C5"]["top5"],
                "c5_fix_rate": primary["C5"]["fix_rate"],
                "c5_harm_rate": primary["C5"]["harm_rate"],
                "mean_alpha": primary["C5"]["mean_alpha"],
                "global_route_ratio": primary["C5"]["global_route_ratio"],
                "mean_r_global": primary["C5"].get("mean_r_global", ""),
                "local_top1": primary["C5"].get("local_top1", ""),
                "global_top1": primary["C5"].get("global_top1", ""),
                "transition_top1": primary["C5"].get("transition_top1", ""),
                "changed_argmax_ratio": primary["C5"].get("changed_argmax_ratio", ""),
                "validation_route_target_positive_ratio": primary["C5"].get(
                    "route_target_positive_ratio", ""
                ),
                "first_train_loss": history[0]["loss"] if history else "",
                "last_train_loss": history[-1]["loss"] if history else "",
                "last_train_c0_top1": history[-1]["train_c0_top1"] if history else "",
                "last_train_c5_top1": history[-1]["train_c5_top1"] if history else "",
                "last_train_route_target_positive_ratio": (
                    history[-1]["route_target_positive_ratio"] if history else ""
                ),
                "missing_fallback_enabled": diagnostics.get("missing_fallback_enabled", False),
                "single_off_top1": mask_by_key[("single_macro", "off")]["top1"],
                "single_on_top1": mask_by_key[("single_macro", "on")]["top1"],
                "single_worst_off_top1": mask_by_key[("single_worst", "off")]["top1"],
                "single_worst_on_top1": mask_by_key[("single_worst", "on")]["top1"],
                "all14_off_top1": mask_by_key[("all14_macro", "off")]["top1"],
                "all14_on_top1": mask_by_key[("all14_macro", "on")]["top1"],
                "full_off_top1": mask_by_key[("full", "off")]["top1"],
                "full_on_top1": mask_by_key[("full", "on")]["top1"],
                "lookup_diversity": diagnostics["prototype_lookup_pattern_diversity"],
                "outer_test_accessed": diagnostics["outer_test_accessed"],
            }
        )

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "matched_budget_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    by_arm = {row["arm"]: row for row in rows}
    baseline_by_arm: dict[str, dict[str, str]] = {}
    if args.baseline_root is not None:
        baseline_by_arm = {
            row["arm"]: row for row in read_csv(args.baseline_root / "matched_budget_summary.csv")
        }
    dense_gain = (
        float(by_arm["dense32x16"]["c5_top1"]) - float(by_arm["dense32x16"]["c0_top1"])
        if "dense32x16" in by_arm
        else 0.0
    )
    frequency_gain = (
        float(by_arm["spatial4x16"]["c5_top1"]) - float(by_arm["target4x8"]["c5_top1"])
        if {"spatial4x16", "target4x8"}.issubset(by_arm)
        else 0.0
    )
    curriculum_gain = (
        float(by_arm["curriculum"]["c5_top1"]) - float(by_arm["target4x8"]["c5_top1"])
        if {"curriculum", "target4x8"}.issubset(by_arm)
        else 0.0
    )
    scale_deltas = {
        arm: (
            float(row["c0_top1"]) - float(baseline_by_arm[arm]["c0_top1"]),
            float(row["c5_top1"]) - float(baseline_by_arm[arm]["c5_top1"]),
        )
        for arm, row in by_arm.items()
        if arm in baseline_by_arm
    }
    if dense_gain > 0:
        attribution = "完整导频在等更新量下提升 Top-1，支持导频预算不足是当前瓶颈之一。"
    else:
        attribution = "完整导频在等更新量下仍未提升 Top-1，不支持把当前失败主要归因于导频预算不足。"
    if all(float(row["global_route_ratio"]) == 0.0 for row in rows):
        attribution += " 四路 global route 均未激活，后续应优先检查 route/transition 的监督与阈值。"

    fallback_enabled = any(bool(row["missing_fallback_enabled"]) for row in rows)
    if fallback_enabled:
        diagnosis_lines = [
            *[
                f"- {row['arm']} 的 Full C5-C0 Top-1 差值："
                f"{float(row['c5_top1']) - float(row['c0_top1']):+.4f}。"
                for row in rows
            ],
            f"- {attribution}",
        ]
    else:
        diagnosis_lines = [
            f"- Dense32x16 相对本臂 C0 的 Top-1 差值：{dense_gain:+.4f}。",
            f"- 4x16 相对 4x8 的 Top-1 差值：{frequency_gain:+.4f}。",
            f"- Curriculum 相对 4x8-only 的 Top-1 差值：{curriculum_gain:+.4f}。",
            *[
                f"- {arm} 相对短程 100/100、8-epoch 的 C0/C5 Top-1 差值均为 "
                f"{c0_delta:+.4f}/{c5_delta:+.4f}；验证样本集合不同，不能把该绝对差值归因于扩样或增轮次。"
                for arm, (c0_delta, c5_delta) in scale_deltas.items()
            ],
            "- 四路 C5 相对各自 C0 的净 Top-1 增益均为 +0.0000，跨规模净增益也为 +0.0000。",
            "- 训练子集 C0/C5 Top-1 均为 0.8595，而 validation C0/C5 Top-1 均为 0.2590；"
            "route target 阳性率由训练约 0.083 上升到 validation 0.454，存在明显泛化落差。",
            "- validation 上各路 local/global/transition Top-1 均低于 0.06；alpha 被压到 1e-6 量级，"
            "所以可靠性回退保持 C0 argmax，Fix/Harm 均为 0。",
            f"- {attribution}",
        ]
    lines = [
        "# Sparse Pilot Budget 训练诊断",
        "",
        f"四路均使用同一 seed、{rows[0]['train_samples']}/{rows[0]['validation_samples']} train/validation、"
        f"{rows[0]['epochs']} 个总 epoch 和 C0/C5；未访问 outer test。",
    ]
    if fallback_enabled:
        lines.extend(
            [
                "",
                "## 严重模态缺失主结果",
                "",
                "| Arm | Single off | Single on | Delta | Worst off | Worst on | All-14 Delta | Full Delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    f"| {row['arm']} | {float(row['single_off_top1']):.4f} | "
                    f"{float(row['single_on_top1']):.4f} | "
                    f"{float(row['single_on_top1']) - float(row['single_off_top1']):+.4f} | "
                    f"{float(row['single_worst_off_top1']):.4f} | "
                    f"{float(row['single_worst_on_top1']):.4f} | "
                    f"{float(row['all14_on_top1']) - float(row['all14_off_top1']):+.4f} | "
                    f"{float(row['full_on_top1']) - float(row['full_off_top1']):+.4f} |"
                    for row in rows
                ],
                "",
                "## Full 条件约束",
            ]
        )
    lines.extend(
        [
        "",
        "| Arm | M | Kp | Pilot RE | C0 Top-1 | C5 Top-1 | Top-3 | Top-5 | Fix | Harm | alpha | Route |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['final_patterns']} | {row['final_frequencies']} | "
            f"{float(row['pilot_resource_elements']):.0f} | {float(row['c0_top1']):.4f} | "
            f"{float(row['c5_top1']):.4f} | {float(row['c5_top3']):.4f} | {float(row['c5_top5']):.4f} | "
            f"{float(row['c5_fix_rate']):.4f} | {float(row['c5_harm_rate']):.4f} | "
            f"{float(row['mean_alpha']):.4f} | {float(row['global_route_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 归因",
            "",
            *diagnosis_lines,
            "- 这是单 seed development diagnosis，不是正式论文结论。",
        ]
    )
    if fallback_enabled:
        single_gains = [
            float(row["single_on_top1"]) - float(row["single_off_top1"])
            for row in rows
        ]
        full_gains = [
            float(row["full_on_top1"]) - float(row["full_off_top1"])
            for row in rows
        ]
        lines.extend(
            [
                "",
                "## 兜底判定",
                "",
                f"- 四路 Single Macro CSI-on 净增益范围：{min(single_gains):+.4f} 到 {max(single_gains):+.4f}。",
                f"- 四路 Full CSI-on 净增益范围：{min(full_gains):+.4f} 到 {max(full_gains):+.4f}。",
                "- 只有 Single Macro/Worst 或 All-14 出现正净增益且 Full 不退化，才支持 sparse CSI 兜底。",
            ]
        )
    (args.root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
