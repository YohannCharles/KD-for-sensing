## Why

当前 Scene32 clean setting 下，image、radar、lidar 等弱模态的全局平均增益很低，但仅凭全局指标无法判断弱模态是完全低效，还是只在局部条件下提供补充信息。需要一个可复现的分析能力，把“强势路径失败、弱模态成功、融合是否 rescue、是否发生负迁移”拆成逐样本证据，支撑后续是否做 sample-wise safe fusion、conditional utility 或弱模态利用机制改造。

## What Changes

- 新增弱模态互补样本分析脚本，从已有 Conditional Utility Audit 产物中读取 subset predictions 和 per-sample delta，生成逐样本 case 表、全局 summary、bucket summary 和模板化报告。
- 新增 schema adapter，兼容 `subset_predictions.csv.gz` / parquet 的真实字段差异，并在概率或 logits 字段缺失时降级为基于 top1 的 case mining。
- 在现有 Gradio 多模态 viewer 中新增 `Complementarity Explorer` / `弱模态互补样本分析` Tab，支持 scene、horizon、弱模态、case type、bucket、排序和阈值筛选。
- 在前端展示当前筛选下的核心研究指标、case 表、统计图和样本详情，并复用现有 raw / processed 多模态展示与预测分布诊断能力。
- 支持导出筛选后的 CSV，便于把互补样本、unused complementary 样本和 negative transfer 样本用于后续实验。
- 增加自动化测试，覆盖 case 判定、summary 指标、缺失概率字段、subset 命名适配和前端筛选逻辑。

## Capabilities

### New Capabilities

- `weak-modality-complementarity-analysis`: 定义弱模态互补样本分析的后端产物、研究指标、Gradio Explorer 交互、降级行为和测试要求。

### Modified Capabilities

- 无。

## Impact

- 影响代码区域：
  - `scripts/analysis/`：新增互补样本分析命令入口。
  - `src/kd_sensing/diagnostics/` 或相邻分析模块：新增 schema 适配、case mining、summary 与筛选逻辑。
  - `tools/visualization/gradio_multimodal_viewer.py` 及 viewer 工具模块：新增 Complementarity Explorer Tab 并复用现有样本详情渲染。
  - `tests/`：新增或扩展分析与 viewer 筛选相关测试。
- 输入依赖：
  - 优先复用 `outputs/scene32/scene32_marf/conditional_utility/subset_predictions.csv.gz`。
  - 可选合并 `conditional_utility_per_sample_delta.csv.gz` 与 `conditional_utility_by_bucket.csv`。
- 输出产物：
  - `outputs/{scene}/complementarity_analysis/complementarity_cases.csv.gz`
  - `outputs/{scene}/complementarity_analysis/complementarity_summary.json`
  - `outputs/{scene}/complementarity_analysis/complementarity_by_bucket.csv`
  - `outputs/{scene}/complementarity_analysis/complementarity_report.md`
- 不引入训练流程行为变更，不改变模型结构、checkpoint、训练日志或 Conditional Utility Audit 的现有输出语义。
