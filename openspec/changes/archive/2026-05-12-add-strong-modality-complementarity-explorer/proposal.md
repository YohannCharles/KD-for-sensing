## Why

当前 Complementarity Explorer 只能从固定 `strong_only` 基线出发筛选 `Weak Modality`，不方便回答“某个强势模态单独失败时，哪个弱模态能补上信息”这类问题。需要在现有弱模态互补分析上增加强势模态维度，让用户能选择一个强势模态，并查看它与一个或全部弱模态的逐样本互补关系。

## What Changes

- 扩展互补分析后端，支持以一个或多个 `strong_modality` 作为 anchor 生成 `strong_modality × weak_modality × horizon × sample` case 表。
- 新增强势模态预测来源选择逻辑，优先使用单模态 subset，其次使用 `teacher_predictions` 中对应强势模态；当前默认强势模态面向 `gps`、`mmwave`，并允许配置覆盖。
- 扩展 case schema 与 summary metadata，新增 `strong_modality`、`strong_prediction_source`、可选 `strong_plus_weak_subset` / `fusion_prediction_available` 等字段，保留现有 `weak_modality` 与 case type 语义。
- 当缺少某个 `strong + weak` fusion subset 时，后端仍生成强势模态与弱模态的互补 case，并将 fusion/rescue 相关指标标记为不可用，而不是中断分析。
- 在 Gradio `Complementarity Explorer` 中新增 `Strong Modality` 控件，行为模仿现有 `Weak Modality` 控件，支持选择单个强势模态或 `all`，并与 scene、horizon、weak modality、case/tag、bucket、gain、sort 筛选联动。
- 更新导出 CSV、样本详情 JSON、统计图和 README，让筛选结果明确展示当前 strong/weak 组合及预测来源。
- 增加自动化测试，覆盖强势模态选择、全部 weak modality 筛选、缺失 fusion subset 降级、summary 分组和 Gradio helper 行为。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `weak-modality-complementarity-analysis`: 扩展现有互补分析与 Explorer，支持按强势模态选择 anchor，并查看该强势模态与一个或全部弱模态的互补关系。

## Impact

- 影响代码区域：
  - `src/kd_sensing/diagnostics/complementarity.py`：扩展 case table 构建、预测来源选择、summary 分组和 metadata。
  - `scripts/analysis/build_complementarity_cases.py`：新增强势模态相关 CLI 参数与日志输出。
  - `tools/visualization/complementarity_explorer.py`：新增 strong modality choices、筛选、统计和导出字段。
  - `tools/visualization/gradio_multimodal_viewer.py`：新增 `Strong Modality` 控件并接入回调。
  - `tools/visualization/README.md`：更新运行与筛选说明。
  - `tests/test_complementarity_analysis.py`、`tests/test_gradio_complementarity_explorer.py`：扩展单元测试。
- 输入依赖：
  - 复用现有 Conditional Utility Audit 的 `subset_predictions`、`teacher_predictions`、`conditional_utility_per_sample_delta` 和 bucket 产物。
  - 对当前 Scene32，`teacher_predictions` 通常可提供 `gps`、`mmwave`、`image`、`radar`、`lidar` 单模态 teacher 预测；若缺失，summary metadata 必须记录不可用原因。
- 不改变训练流程、模型结构、checkpoint 或已有 `strong_only + weak` 互补分析默认行为。
