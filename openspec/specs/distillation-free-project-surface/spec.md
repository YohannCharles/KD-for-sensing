# distillation-free-project-surface Specification

## Purpose

约束 T2/baseline 不再暴露独立蒸馏运行面，同时明确同一 primary model 内部的 T2 consistency、BPA 和 CMA 不属于外部 teacher 工作流。

## Requirements

### Requirement: 项目表面不再提供蒸馏能力

系统 MUST 不提供独立 teacher-student runtime、外部 teacher tensor/checkpoint、full-to-partial KD、weak-pattern KD 或 legacy KD compatibility。T2 可保留同一 primary model 的 online no-grad full/superset consistency、embedded teacher CE、BPA/prototype 与 active CMA ablation，且它们 MUST 不构建第二模型或读取 teacher artifact。

#### Scenario: 运行 T2 或 S1

- **WHEN** 用户启动 T2 或 S1 recipe
- **THEN** runtime MUST 只构建被优化的 primary model
- **AND** 不得读取外部 teacher checkpoint 或 tensor

### Requirement: 退役蒸馏字段不映射

current config、registry、CLI 和文档 MUST 不提供 retired distillation field 的 fallback 或迁移。

#### Scenario: 提供旧蒸馏字段

- **WHEN** 用户在 current recipe 中声明退役字段
- **THEN** validation MUST 拒绝或忽略该未知字段
- **AND** 系统 MUST 不生成替代 teacher branch
