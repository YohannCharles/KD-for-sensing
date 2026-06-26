## ADDED Requirements

### Requirement: Attention provenance 与归一化可审计
JEPA visual analysis MUST 为 GPS-query attention 图和 attention summary 输出可审计 provenance。系统 MUST 区分 token-read map、gradient/ablation attribution 和其它 saliency 方法，并 MUST 记录图表是否跨 head、time、query 或 sample 归一化。

#### Scenario: attention provenance 写入 manifest
- **WHEN** JEPA visual analysis 导出 attention patch-grid、query-time panel 或 image overlay
- **THEN** `analysis_manifest.json` MUST 记录 map semantics、attention source module、attention tensor shape、token grid、aggregation method、normalization scope 和 overlay image source
- **AND** report MUST 不把 `token_read_map` 单独描述为 causal attribution

#### Scenario: per-sample minmax 标记不可跨样本比较
- **WHEN** attention overlay 使用 per-sample minmax 或 per-sample shared minmax 归一化
- **THEN** manifest 或 overlay row MUST 记录该归一化不支持跨样本强度比较
- **AND** report MUST 将跨样本 attention 强度比较标记为不可用或 exploratory

#### Scenario: 全局归一化可选
- **WHEN** analysis config 启用全局或数据集级 attention 归一化
- **THEN** 系统 MUST 使用同一模型和同一分析批次的共享 attention 范围生成可比 overlay
- **AND** manifest MUST 记录用于共享尺度的样本数、min/max 或分位数范围

### Requirement: JEPA attention faithfulness 输出
JEPA visual analysis MUST 支持 opt-in attention faithfulness 诊断，用于检查 GPS-query token-read map 与模型预测敏感性是否一致。该诊断 MUST 使用 deterministic patch selection 和遮挡策略，并 MUST 在缺少输入、attention 或 optional dependency 时安全降级。

#### Scenario: 运行 attention faithfulness
- **WHEN** analysis config 启用 attention faithfulness 且模型提供 attention map
- **THEN** 系统 MUST 对选中样本运行 top-attention、low-attention 和 random patch 遮挡诊断
- **AND** 系统 MUST 写出 `tables/attention_faithfulness.csv`
- **AND** `analysis_manifest.json` MUST 记录 occlusion strategy、patch budget、seed、metric target 和 skipped reasons

#### Scenario: 导出 faithfulness 图表
- **WHEN** attention faithfulness 表至少包含一个有效模型和样本
- **THEN** 系统 MUST 在可视化依赖可用时导出简洁图表到 `figures/attention_faithfulness/`
- **AND** 图表 MUST 标注模型、metric、patch selection group、patch ratio 和样本数
- **AND** 可视化依赖不可用时系统 MUST 写出 skipped reason 并保留 CSV 产物

#### Scenario: faithfulness 不影响主分析完成
- **WHEN** attention faithfulness 诊断因缺少 attention、image tensor、raw image 或可遮挡 token fallback 而不可运行
- **THEN** JEPA visual analysis MUST 继续生成 logits metrics、attention summary、case study 和 report 中其它可用产物
- **AND** 系统 MUST 在 `analysis_manifest.json` 和 `report.md` 中记录 `attention_faithfulness_unavailable`

### Requirement: Report 区分性能、解释和失败模式
JEPA visual analysis report MUST 将 GPS-query attention 相关发现拆分为性能证据、解释性诊断和失败模式。报告 MUST 引用 paired metrics、faithfulness 表、attention summary 和 case selection，而不得只展示成功样本或单张 overlay。

#### Scenario: 报告 attention faithfulness
- **WHEN** attention faithfulness 诊断可用
- **THEN** `report.md` MUST 汇总 top-attention、low-attention 和 random 遮挡的平均 metric delta
- **AND** report MUST 说明该结果是否支持 token-read map 的解释性使用

#### Scenario: 报告失败或退化样本
- **WHEN** case selection 或 faithfulness 诊断发现 query regression、shared failure 或 attention 不可信样本
- **THEN** report MUST 在 caveat 或失败模式段落列出这些样本组
- **AND** report MUST 不只展示 query gain 或 visually plausible overlay
