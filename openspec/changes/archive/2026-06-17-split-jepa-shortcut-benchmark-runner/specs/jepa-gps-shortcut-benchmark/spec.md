## ADDED Requirements

### Requirement: Benchmark runner 内部模块化
JEPA GPS shortcut benchmark runner SHALL 将 manifest/schema、suite-specific perturbation normalization、metric aggregation、artifact writing 和 plotting 拆分到职责明确的内部模块。原 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` MUST 保留为公开 facade，并 MUST 不承载新增 suite-specific helper 实现。

#### Scenario: 公开 facade 保持兼容
- **WHEN** 现有代码从 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` 导入公开 runner 或 analysis bundle helper
- **THEN** 导入 MUST 继续成功
- **AND** CLI `kd-sensing-jepa-gps-shortcut-benchmark` MUST 继续调用同一公开语义

#### Scenario: suite helper 不回流 facade
- **WHEN** 新增或修改 Scenario C、Scenario D、CxD 或 Predictive JEPA helper
- **THEN** 主要实现 MUST 位于对应窄模块
- **AND** facade MUST 只做兼容导出、薄 orchestration 或向后兼容包装

### Requirement: Benchmark 输出 schema 保持兼容
拆分 JEPA GPS shortcut benchmark runner MUST 保持现有 manifest 输入、输出文件名、CSV/JSON 字段、warnings schema 和 visual-analysis ingestion bundle 兼容。任何输出 schema 变更 MUST 通过单独 OpenSpec change 明确声明。

#### Scenario: smoke manifest 输出字段稳定
- **WHEN** 测试运行 smoke 或 mock benchmark manifest
- **THEN** 输出 `benchmark_manifest.json`、`metrics_by_condition.csv` 和 `robustness_summary.csv` 的必需字段 MUST 与拆分前兼容
- **AND** mock/smoke 行 MUST 继续标记为 mock、smoke 或 unavailable，不得冒充真实数值 claim

#### Scenario: Scenario D/CxD artifact 保持可消费
- **WHEN** Scenario D 或 CxD smoke manifest 完成
- **THEN** runner MUST 继续写出现有 Scenario D/CxD result tables、heatmap artifacts 和 phase/dominance/crossing metadata
- **AND** `kd-sensing-jepa-visual-analysis` MUST 能继续只读消费 runner manifest
