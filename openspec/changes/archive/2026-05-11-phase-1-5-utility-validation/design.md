## Context

当前代码已经完成 Phase 1 的主体能力：`src/kd_sensing/evaluation/subset_specs.py` 定义了 Scene32 conditional audit subset，`tools/analysis/run_conditional_utility_audit.py` 能从 MARF checkpoint 生成 `subset_predictions.csv.gz`、`conditional_utility_per_sample_delta.csv.gz`、`teacher_predictions.csv.gz`、bucket CSV、oracle summary、teacher complementarity summary 和 figures。

已有 `outputs/scene32/scene32_marf/conditional_utility/` 结果显示：

- `strong_only` 平均 Top1/Top3/DBA 为 `0.423960 / 0.780241 / 0.903695`。
- `all` 平均 Top1/Top3/DBA 为 `0.423182 / 0.778296 / 0.903617`，略低于 `strong_only`。
- `strong_plus_image/radar/lidar` 相对 `strong_only` 的平均 DBA delta 只有 `+0.00031 / +0.00031 / +0.00023`，Top1/Top3 没有稳定提升。
- teacher rescue rate 约 `6.48%`，oracle DBA gain 约 `+0.00093`。

这些结果支持“Scene32 clean setting 下弱模态边际效用很低或存在负迁移”的初步判断，但仍不足以排除两个可能性：第一，极小增益只是统计波动；第二，当前 MARF masking path 没有利用弱模态，但 dedicated fixed-subset 训练可能能学到收益。

## Goals / Non-Goals

**Goals:**

- 对现有 Phase 1 逐样本结果补 paired cluster bootstrap 95% CI，判断微小 delta 是否显著。
- 对 `scene32_marf` 的 `best_top1.pth`、`best.pth` 和 `last.pth` 重跑或汇总 audit，检查结论是否依赖 checkpoint 选择。
- 训练 dedicated fixed-subset baseline：`gps+mmwave`、`gps+mmwave+image`、`gps+mmwave+radar`、`gps+mmwave+lidar`、`all`，至少 3 seeds。
- 生成统一 Phase 1.5 report，输出 mean/std、CI、per-horizon 指标、bucket/horizon 条件性证据和路线建议。
- 将 diagnosis 从“delta 为正即可解释”改成“达到阈值且置信区间支持”的规则。

**Non-Goals:**

- 不设计或实现 MARF-v2、MARF-Comm、新 router 输入、GPS-conditioned image mask、sector attention、BEV alignment 或跨模态交互模块。
- 不修改现有 MARF 主结构、loss 目标、encoder 冻结策略、训练默认行为或 Phase 1 audit 默认输出。
- 不把弱模态 anchor 权重、prior bias 或 residual 策略作为 Phase 1.5 的调参目标。
- 不把 legacy 单 seed best checkpoint 直接当作最终 dedicated baseline 结论，除非 metadata 满足同预算、同选择规则和同 seed matrix 的可比性要求。

## Decisions

1. Phase 1.5 作为独立分析/实验编排层实现。

   方案：新增 Phase 1.5 脚本和报告目录，复用现有训练入口、canonical fusion config、audit runner 和逐样本表。普通训练、评估、Phase 1 audit 不默认运行 Phase 1.5。

   备选：把 bootstrap 和多 checkpoint 逻辑塞进 `run_conditional_utility_audit.py`。拒绝原因是 audit runner 应继续负责单 checkpoint 产物，Phase 1.5 负责跨 run 汇总和决策。

2. bootstrap 使用 paired delta 表作为主输入。

   方案：对 `conditional_utility_per_sample_delta` 中每个 `weak_modality + horizon` 的 `delta_ce`、`delta_top1`、`delta_top3`、`delta_dba` 做重采样；`all - strong_only` 从 `subset_predictions` 现场配对计算。若 `seq_id` 不存在，使用 `sample_id` 或 `dataset_index` 作为 cluster key，并在输出 metadata 中声明 fallback。

   备选：直接对 aggregate metrics 做无配对 bootstrap。拒绝原因是 subset 之间共享同一批样本，paired delta 能降低噪声并保持比较语义。

3. dedicated baseline 采用现有 fusion teacher/student 训练能力，而不是新增模型。

   方案：使用 canonical modality slug 和命令行覆盖生成 5 个 fixed-subset no-KD 训练。Scene32 模态顺序保持 `image -> radar -> gps -> lidar -> mmwave`，例如 `gps_mmwave_teacher_no_kd`、`image_gps_mmwave_teacher_no_kd`、`radar_gps_mmwave_teacher_no_kd`、`gps_lidar_mmwave_teacher_no_kd`、`image_radar_gps_lidar_mmwave_teacher_no_kd`。

   备选：在 MARF 内用 mask 训练固定子路径。拒绝原因是这仍会受 MARF anchor/router 结构影响，不能回答“专门训练的 subset 是否有价值”。

4. baseline 比较以 dedicated `gps+mmwave` 为主基线。

   方案：所有 strong+weak 和 all 的显著性判断都以同 seed、同预算训练得到的 dedicated `gps+mmwave` 为基线，同时保留当前 MARF masking `strong_only` 作为 Phase 1 诊断参照。

   备选：继续用 MARF masking `strong_only=0.4240` 作为最终基线。拒绝原因是该指标只是当前 MARF 子路径，不代表专门训练后的强模态上限。

5. checkpoint matrix 固定为三类角色，而非任意文件列表。

   方案：每个 MARF run 至少比较 `best_top1.pth`、`best.pth` 和 `last.pth`；如果项目后续显式保存 `best_dba.pth`，Phase 1.5 runner 可优先用 `best_dba.pth` 替代或补充 `best.pth`，并在 manifest 中记录 checkpoint role。

   备选：只重跑当前默认 checkpoint。拒绝原因是 Phase1.5.md 已指出 epoch 75 与 epoch 100 可能有不同 router 塌缩程度。

6. decision gate 明确服务后续路线选择。

   方案：报告输出 `low_weak_utility`、`conditionally_useful`、`representation_exists_but_not_exploited`、`safe_fusion_candidate` 等标签，但最终推荐只分两条路线：无显著收益则转向 strong-path + safe fusion + degraded robustness；有稳定 bucket/horizon 收益才进入 MARF-Comm。

   备选：只给表格不做路线判断。拒绝原因是 Phase 1.5 的核心价值是防止继续投入证据不足的复杂融合方向。

## Risks / Trade-offs

- cluster key 不完整 -> 使用 `sample_id/dataset_index` fallback，并在 report 中明确这不是严格 seq-level bootstrap。
- 3 seeds x 5 baseline 训练成本较高 -> 允许先生成 run manifest 和命令矩阵，训练产物缺失时 report 标记 `pending`，但不得给最终结论。
- 现有历史 baseline 已有单 seed 输出但协议不完全一致 -> 可作为参考列展示，不能进入 final mean/std 和显著性判定。
- bootstrap 对高度相关样本仍可能乐观 -> 首选 seq-level cluster；如果无法解析 `seq_id`，结论措辞必须降级为“当前 sample-level evidence”。
- CI 阈值可能过严导致局部信号被忽略 -> bucket/horizon 结果仍保留为探索性证据，但 MARF-Comm 入口要求可重复 checkpoint 或 dedicated baseline 支持。
- 多 checkpoint audit 会重复推理时间 -> 复用现有 audit runner 输出目录，并在 manifest 中跳过已完成且 metadata 匹配的产物。

## Migration Plan

1. 新增 Phase 1.5 统计 helper，先能读取已有 `outputs/scene32/scene32_marf/conditional_utility/` 并生成 bootstrap CI。
2. 新增 checkpoint matrix manifest 和 runner，针对 `best_top1.pth`、`best.pth`、`last.pth` 调用现有 audit runner。
3. 新增 fixed-subset baseline manifest 和命令生成/汇总逻辑，要求 5 subsets x 3 seeds 的训练与评估产物齐全后才生成最终结论。
4. 新增 Phase 1.5 report writer，汇总 bootstrap、checkpoint matrix、baseline matrix、bucket highlights 和 decision gate。
5. 更新 conditional utility diagnosis，使配置启用 CI 时按 CI 下界与最小 delta 阈值判定。
6. 定向验证：`conda run -n kd_mm_beam pytest -q tests/test_phase_1_5_utility_validation.py tests/test_conditional_utility_metrics.py`；必要时再运行相关 analysis script 的 smoke test。

## Open Questions

- 当前 dataset metadata 是否能可靠恢复 `seq_id`。现有 Phase 1 输出包含 `sample_id`、`dataset_index` 和路径，但没有显式 `seq_id`；首版需要决定是否从路径/CSV 补解析，或先声明 fallback。
- 现有训练入口是否已经支持所有 5 个 fixed-subset slug 的命令行 seed/run_name 覆盖；若不支持，需要补最小配置生成器或命令矩阵脚本。
- `best.pth` 在当前训练流程中更接近 best loss 还是 best DBA。Phase 1.5 report 必须读取 checkpoint metadata 或 metrics 说明 checkpoint role。
