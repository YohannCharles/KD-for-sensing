## MODIFIED Requirements

### Requirement: 项目表面不再提供蒸馏能力
系统 MUST 不提供独立 teacher-student distillation 作为源码、配置、注册表、训练运行时和文档支持能力。新训练、评估、诊断和 summary 输出 MUST 不构建 frozen/checkpoint teacher、不实例化 distiller、不恢复旧 `distillation.*`。唯一允许的窄例外是 current U-Mask 或 active OpenSpec 明确批准的同一 primary model 在线 stop-gradient full/superset consistency；该例外 MUST 通过方法 training extension 表达，MUST 不引入第二个模型实例或 teacher checkpoint。

#### Scenario: 新训练运行没有独立 distillation runtime
- **WHEN** 用户运行任一当前支持的训练配置
- **THEN** 系统 MUST 只构建被优化的 primary model
- **AND** 系统 MUST 不构建 frozen teacher model
- **AND** 系统 MUST 不实例化 distiller、RKD 或 legacy teacher-student runtime

#### Scenario: Same-model online superset 例外
- **WHEN** active current config 显式启用 same-model online full/superset consistency
- **THEN** teacher output MUST 由 primary model 的 no-grad forward 产生
- **AND** evaluation MUST 不执行该 teacher branch
- **AND** config/metadata MUST 使用方法 owner 的 `superset_consistency` 或既有 U-Mask stabilization 字段，不得使用 `distillation.*`、`teacher_checkpoint` 或 `legacy_kd`

#### Scenario: 新产物没有 legacy KD metadata
- **WHEN** 训练或评估写出 `final_config.yaml`、run metadata、train log、TensorBoard scalar 或 summary artifact
- **THEN** 新产物 MUST 不包含 `distillation_enabled`、`distillation_type`、`teacher_checkpoint`、`teacher_source` 或 `legacy_kd` lifecycle 字段
- **AND** loss 字段 MUST 使用 `loss/beam`、`loss/primary`、`loss/superset_consistency`、`loss/beam_monotonic_rank` 或 workflow-specific supervised/adaptation 命名
