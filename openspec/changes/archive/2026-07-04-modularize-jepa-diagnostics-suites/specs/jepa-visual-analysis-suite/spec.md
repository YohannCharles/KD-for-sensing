## ADDED Requirements

### Requirement: JEPA visual analysis 必须按 artifact owner 模块化
JEPA visual analysis 实现 MUST 将 config loading、model analysis、table writing、figure writing、case payload generation、evidence package integration、report building 和 manifest building 拆分到清晰 owner helper 或模块，并保持 public runner 行为兼容。

#### Scenario: visual analysis 输出 schema 兼容
- **WHEN** 用户运行 `kd-sensing-jepa-visual-analysis` 或包内 runner
- **THEN** 输出目录 MUST 根据启用配置继续包含兼容的 `analysis_manifest.json`、`report.md`、tables、figures、cache metadata 和 case payloads
- **AND** optional model failures MUST remain recorded as warnings instead of silently dropping models

#### Scenario: 新图表不扩大主 runner
- **WHEN** 新增 JEPA visual analysis figure or table
- **THEN** main runner MUST 委托 focused writer/helper
- **AND** focused tests MUST 覆盖新增 artifact registration 或 skipped-output 行为
