## Why

现有 P0-P5 结果对 Image-only、GPS-only 和 Fusion 大类有区分，但对不同融合模型的诊断区分度不足：P1/P2/P5 主要测图像扰动，P3/P4 主要测 GPS 错误或混合强扰动，导致融合模型趋势高度一致。现在需要一套复用现有模型权重的更小、更正交的指标包，用来判断模型是否真的会在 GPS 与图像可靠性变化时做有效融合。

## What Changes

- 新增一个复用权重的融合诊断评估方案：输入现有 model config、checkpoint、split、seed 和 metric profile，不重新训练。
- 评估一个小型正交切片，而不是继续只看 P0-P5 平均：
  - clean anchor：`C0_sync + D0_full_image`
  - 图像受损、GPS 正常：如 `C0_sync + D4_partial_occlusion`、`C0_sync + D6_burst_missing`
  - GPS 异步/缺失、图像正常：如 `C3_random_async + D0_full_image`、`C4_severe_async + D0_full_image`
  - 双模态受损：如 `C3_random_async + D4_partial_occlusion`、`C4_severe_async + D6_burst_missing` 或 `C4_severe_async + D7_joint_worst_case`
  - hard negative：复用 GPS-query advantage slice 中的 visual ambiguity 与 beam-offset-constrained wrong GPS 条件。
- 输出 condition-level DBA/Top-K、相对 clean drop、paired baseline margin，以及 `image_rescue`、`gps_rescue`、`fusion_interaction` 等派生指标。
- 修改 Predictive Robustness 的 claim 解释：P0-P5 保留为兼容/鲁棒性表，但融合机制主诊断必须优先看正交 CxD/A-slice 指标。
- 不新增依赖，不提交 `outputs/`、cache、checkpoint 或日志产物。

## Capabilities

### New Capabilities
- `reused-weight-fusion-diagnostic-metrics`: 复用现有模型权重运行 CxD 小切片、GPS-query hard-negative 切片和融合派生指标的离线诊断能力。

### Modified Capabilities
- `predictive-jepa-robustness`: 调整 claim 口径，要求 P0-P5 之外的正交 CxD/A-slice 证据用于融合机制区分；P0-P5 不再单独作为融合机制主证据。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/` 中 JEPA GPS shortcut benchmark、manifest normalization、metrics aggregation 和报告输出。
- 可复用现有 `scenario_c_x_d_image_observability`、`scenario_d_image_observability`、`predictive_jepa_robustness` 和 GPS-query advantage slice 逻辑。
- 需要新增或扩展 focused tests，覆盖 manifest 解析、复用权重 real-forward 配置、派生指标计算和 claim gate 文案。
- 运行产物写入 ignored `outputs/analysis/...`，源码只包含 OpenSpec、实现、测试和必要配置示例。
