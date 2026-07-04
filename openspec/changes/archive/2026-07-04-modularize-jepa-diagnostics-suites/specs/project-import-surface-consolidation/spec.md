## ADDED Requirements

### Requirement: JEPA diagnostics 内部导入必须直连 owner
Internal diagnostics code MUST 从具体 owner 模块导入 JEPA benchmark 和 visual analysis helper。Public facades MAY 只供 CLI glue 和已文档化 public import path 使用。

#### Scenario: 内部 facade 回流被拒绝
- **WHEN** diagnostics, engine, data, models, losses or ordinary tests import private benchmark helpers from `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark`
- **THEN** architecture boundary checks MUST 失败
- **AND** the failure MUST point to the corresponding `jepa_benchmark_*` owner module
