## ADDED Requirements

### Requirement: CSI hardening 配置命名去 KD 化
CSI degradation 和 hardening 配置 MUST 使用 supervised、strong、lightweight 或 workflow-specific 命名，不得通过 `no_kd` 或 distillation block 表达普通 supervised baseline。

#### Scenario: CSI supervised 配置可加载
- **WHEN** 用户加载 CSI degradation 或 hardening supervised 配置
- **THEN** 最终配置 MUST 不包含 `distillation`
- **AND** 输出 run name MUST 不包含 `_no_kd`

