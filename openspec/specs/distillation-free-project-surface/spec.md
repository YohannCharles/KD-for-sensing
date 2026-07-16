# distillation-free-project-surface Specification

## Purpose

约束 T2/baseline 不再暴露独立蒸馏运行面，同时明确同一 primary model 内部的 T2 consistency、BPA 和 CMA 不属于外部 teacher 工作流。

## Requirements

### Requirement: 项目表面不再提供蒸馏能力

系统 MUST 不提供独立 teacher-student distillation、外部 teacher tensor/checkpoint guidance、full-to-partial KD、weak-pattern KD 或 legacy KD compatibility 作为源码、配置、注册表、训练运行时和文档支持能力。唯一保留的是 T2 同一 primary model 的在线 no-grad full/superset consistency、embedded full-modal teacher CE、BPA/prototype 及 active CMA ablation；这些机制 MUST 不构建第二模型或读取 teacher checkpoint。

#### Scenario: T2 只构建 primary model

- **WHEN** 用户运行任一 T2 或 S1 recipe
- **THEN** runtime MUST 只构建被优化的 primary model
- **AND** teacher logits MUST 只来自该模型的同次在线 no-grad forward

#### Scenario: 外部 teacher 支线不存在

- **WHEN** 配置声明 `teacher_guidance`、`teacher_checkpoint`、`full_to_partial_kd`、`weak_pattern_kd` 或等价 legacy field
- **THEN** 系统 MUST 不提供对应实现或兼容映射
- **AND** 当前 recipe MUST 不生成这些字段

#### Scenario: 评估不执行 teacher branch

- **WHEN** 用户评估 T2 或 baseline checkpoint
- **THEN** evaluation MUST 不执行 online teacher consistency forward
- **AND** 输出 metadata MUST 不包含 legacy teacher artifact provenance

### Requirement: 退役蒸馏字段不映射

current config、registry、CLI 和文档 MUST 不提供 retired distillation field 的 fallback 或迁移。

#### Scenario: 提供旧蒸馏字段

- **WHEN** 用户在 current recipe 中声明退役字段
- **THEN** validation MUST 拒绝或忽略该未知字段
- **AND** 系统 MUST 不生成替代 teacher branch
