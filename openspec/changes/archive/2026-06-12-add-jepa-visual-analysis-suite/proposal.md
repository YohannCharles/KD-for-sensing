## Why

当前 GPS-query JEPA pooling 在 2604-style S32/S33/S34 评估上已经取得更高 DBA，但还缺少一套可复现、可审稿的机制解释证据链。我们需要把“JEPA 为什么更好”从单一指标扩展为表示空间、注意力聚合、错误邻近性、样本案例和鲁棒性切片等可视化分析，支撑论文主张并避免把提升解释成偶然 split 或过拟合。

## What Changes

- 新增 JEPA 可视化分析套件，用固定模型 checkpoint 和固定评估 split 导出论文级图表与 JSON/CSV 摘要。
- 支持对比至少三类模型：`fair_base`、`fair_gps_biased`、`gps_query_pool`，并允许扩展到更多 baseline。
- 导出表示空间可视化：UMAP/t-SNE、按 beam/scene/error bucket 着色、邻近 beam 结构指标。
- 导出预测行为可视化：Top-k rank transition、Top-1 error histogram、Top-3 min-distance histogram、DBA contribution 分布、beam confusion/near-miss matrix。
- 导出 GPS-query attention 与图像显著性可视化：query-to-patch attention heatmap、时间帧 attention drift、典型成功/失败 case overlay。
- 导出鲁棒性切片：drop image、drop GPS、GPS noise、image masking 或复用已有 modality subset 评估结果，展示 JEPA 表征是否更稳定。
- 新增统一报告 manifest，记录输入 checkpoint、config、split、seed、样本选择规则、图表路径和关键数字，保证结果可追溯。
- 不引入破坏性变更；训练、评估、viewer 现有入口继续保持兼容。

## Capabilities

### New Capabilities

- `jepa-visual-analysis-suite`: 定义 JEPA 下游 beam prediction 的可复现可视化分析、对比报告、样本导出、机制解释图表和鲁棒性切片产物。

### Modified Capabilities

- 无。

## Impact

- 影响代码范围：新增或扩展 `src/kd_sensing/diagnostics/` 下的分析模块，新增一个 CLI 入口或脚本用于批量导出图表与报告。
- 影响实验产物：新增 `outputs/visual_analysis/<run_id>/` 风格的本地分析目录，包含 PNG/SVG/PDF、CSV、JSON summary 和 manifest；默认不纳入源码提交。
- 影响配置：可新增 `configs/diagnostics/` 或实验配置片段，用于声明模型对比组、checkpoint 路径、评估 split、采样策略和图表开关。
- 影响依赖：优先复用现有 `matplotlib`、`numpy`、`torch`、`pandas`、`sklearn`；如需 UMAP，作为可选依赖或自动降级到 PCA/t-SNE，不强制影响核心训练依赖。
- 影响验证：新增单元测试覆盖指标计算、manifest schema、样本选择稳定性、无 checkpoint 的 dry-run，以及小型 synthetic logits 的图表导出路径。
