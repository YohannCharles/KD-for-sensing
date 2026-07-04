## ADDED Requirements

### Requirement: 低风险瘦身必须与行为重构隔离
机械 import 清理、内部 export 清理和测试文件拆分 MUST 与 training/model/data 行为变更分开规划。

#### Scenario: mechanical cleanup 独立验证
- **WHEN** future imports, star imports or internal `__all__` entries are removed
- **THEN** tasks MUST run architecture/import focused checks
- **AND** implementation notes MUST state that no runtime behavior change was intended
