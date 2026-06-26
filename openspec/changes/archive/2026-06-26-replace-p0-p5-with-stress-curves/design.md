## Context

现有 Predictive Robustness 使用 P0-P5 离散条件表达图像缺失、图像遮挡、wrong GPS、joint recovery 和天气扰动。实际分析中，`P1/P2/P5` 与 clean 或彼此的区分度不足，`P4_joint_predictive_recovery` 又把当前帧缺失、随机遮挡和 wrong GPS 叠在一起：当前帧已经缺失时再遮挡没有额外评估价值，反而让结果不可解释。

仓库已经有统一 difficulty pipeline、Scenario C GPS async、Scenario D image observability、GPS-query advantage slice 和 benchmark 输出聚合。最小改法是复用这些机制，把 predictive 主评估从“六个命名档位”改为“少数 stress suite 的 severity sweep”。

## Goals / Non-Goals

**Goals:**

- 用 clean anchor + 单轴 stress curves 测模型抗干扰上限。
- 主评估只保留三类默认 stress：`image_missing`、`image_noise`、`gps_noise`。
- 每条曲线输出 clean delta、relative retention、`S@drop<=0.02`、`S@drop<=0.05`、`AUC_retention` 和 `collapse_s`。
- 明确 `joint_stress` 只是可选二阶段诊断，不参与默认主 claim。
- 保留 strict comparability、difficulty digest、seed、split、sample_count 和产物边界。

**Non-Goals:**

- 不新增模型结构、训练 loss 或 checkpoint 格式。
- 不新增外部依赖。
- 不恢复旧 P0-P5 heatmap 作为主 claim 图。
- 不把 joint stress 设计成新的大而全 claim gate。

## Decisions

1. 默认主评估使用三条单轴曲线，而不是 P0-P5 离散表。

   - 选择：`image_missing`、`image_noise`、`gps_noise`。
   - 原因：每条曲线只回答一个问题：缺失能扛到多少、视觉扰动能扛到多少、GPS 扰动能扛到多少。
   - 替代方案：继续修补 P0-P5。拒绝原因是低区分度条件会继续稀释结论，尤其 P4 的缺失+遮挡叠加本身无效。

2. `image_missing` 表达为窗口内缺失比例或末端连续缺失比例。

   - 默认先使用末端连续缺失：`missing_tail_fraction`，例如最后 25%、50%、75%、100% 图像帧缺失。
   - 原因：预测任务最关心当前帧及其临近历史缺失时模型是否能用更早历史或 GPS 兜底。
   - 缺失表达继续保持固定 tensor shape，用 `image_valid_mask=false`、`image_observability_score=0` 和 zero-fill/sentinel 记录。

3. `image_noise` 只选择一种主视觉干扰，不混合天气、遮挡、模糊。

   - 默认建议用 occlusion ratio 或 corruption severity 的单参数 sweep。
   - 原因：一条曲线必须有单调、可解释的 severity 轴；多种视觉增强混在一起会重复旧 P-suite 的问题。
   - 实现时可先复用现有 image occlusion operator；天气/模糊可作为 future suite，而不是默认主轴。

4. `gps_noise` 使用一种默认 GPS 干扰轴，并允许 manifest 显式选择。

   - 默认建议 `gps_jitter_std` 或 beam-offset-constrained wrong-peer ratio 二选一。
   - 更偏工程测量时用 jitter/delay；更偏 hard-negative 诊断时用 wrong peer。
   - 不在默认配置里同时混合 jitter、delay、wrong peer。

5. 主汇总从“P0-P5 mean”切换为曲线级上限指标。

   - `retention = metric(severity) / metric(clean)`。
   - `S@drop<=x` 是满足 `metric(clean)-metric(severity) <= x` 的最大 severity。
   - `AUC_retention` 是 severity-retention 曲线的归一化面积。
   - `collapse_s` 是首次跌破 `metric(clean)-0.10` 的 severity；若未跌破则为空或 `not_collapsed`。
   - `weakest_axis` 是 collapse 最早或 AUC 最低的 stress suite。

6. `joint_stress` 作为二阶段可选诊断。

   - 默认不进入主 claim。
   - 只有在三条单轴曲线跑通并可解释后，才允许用同一 severity 同步组合 `image_missing` 和 `gps_noise`。
   - joint 输出必须标注为 diagnostic，不能替代单轴上限。

## Risks / Trade-offs

- [Risk] 旧报告、配置或测试依赖 P0-P5 字段名。→ Mitigation: 在迁移期允许 legacy P-level 输入被标记为 deprecated/legacy 输出，但新 claim gate 只读 stress suite 指标。
- [Risk] 不同模型 clean metric 很低时 retention 比例失真。→ Mitigation: 同时输出 absolute clean delta 和 retention；clean metric 低于阈值时标记为 `clean_anchor_unstable`。
- [Risk] severity 单位跨 suite 不可比。→ Mitigation: `overall_robustness_score` 只作为粗摘要；报告必须保留逐 suite 曲线和 severity_unit。
- [Risk] GPS wrong peer 与 GPS jitter 回答的问题不同。→ Mitigation: manifest 必须声明 `gps_noise_mode`；默认只启用一种，另一种作为显式可选。
- [Risk] 直接删除旧 P0-P5 可能影响历史结果复现。→ Mitigation: 历史 HTML/CSV 保持本地产物；源码契约中把旧 P-level 标为非主评估或 legacy，而不是要求重写历史产物。

## Migration Plan

1. 新增/调整 predictive stress suite normalization：clean + `image_missing`、`image_noise`、`gps_noise`，可选 `joint_stress`。
2. 将旧 P-level 默认 preset 替换为 stress preset；legacy P0-P5 只保留为显式兼容路径或 deprecated alias。
3. 更新 benchmark aggregation，输出 curve rows 与 stress summary。
4. 更新 configs/docs/tests，删除以 P0-P5 mean 为主 claim 的断言。
5. 使用 synthetic batch focused tests 验证扰动 determinism、mask 语义、summary 指标和 legacy 降级提示。

## Open Questions

- `gps_noise` 默认轴最终选 jitter/delay，还是 beam-offset wrong peer？建议先选 jitter/delay 作工程抗噪上限，wrong peer 留给 hard-negative 诊断。
- `image_noise` 默认轴选 occlusion ratio，还是统一 corruption severity？建议先选 occlusion ratio，最直观且已有 operator 支撑。
