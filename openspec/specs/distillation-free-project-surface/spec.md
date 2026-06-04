# distillation-free-project-surface Specification

## Purpose
TBD - created by syncing change remove-distillation-support. Update Purpose after archive.
## Requirements
### Requirement: 项目表面不再提供蒸馏能力
系统 MUST 删除 teacher-student distillation 作为源码、配置、注册表、训练运行时和文档支持能力。新训练、评估、诊断和 summary 输出 MUST 不构建 frozen teacher、不计算 KD loss、不写出 distillation metadata。

#### Scenario: 新训练运行没有 distillation runtime
- **WHEN** 用户运行任一当前支持的训练配置
- **THEN** 系统 MUST 只构建被优化的 primary model
- **AND** 系统 MUST 不构建 frozen teacher model
- **AND** 系统 MUST 不实例化 distiller、logits KD、RKD 或等价 teacher-student loss

#### Scenario: 新产物没有 KD metadata
- **WHEN** 训练或评估写出 `final_config.yaml`、run metadata、train log、TensorBoard scalar 或 summary artifact
- **THEN** 新产物 MUST 不包含 `distillation_enabled`、`distillation_type`、`teacher_checkpoint`、`teacher_source` 或 `legacy_kd` lifecycle 字段
- **AND** loss 字段 MUST 使用 `loss/beam`、`loss/primary`、`loss/beam_soft_target` 或 workflow-specific supervised/adaptation 命名
### Requirement: 旧 KD 入口必须清晰拒绝
系统 MUST 拒绝 `logits_kd`、`rkd`、`distillation.*`、`*_no_kd` legacy config path 和旧 distiller registry 名称。拒绝 MUST fail fast，并给出当前 supervised、strong 或 lightweight 入口建议。

#### Scenario: 旧 KD 配置路径被拒绝
- **WHEN** 用户加载 `configs/<modality>/logits_kd.yaml`、`configs/<modality>/rkd.yaml`、`configs/fusion/<slug>_logits_kd.yaml` 或 `configs/fusion/<slug>_rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 说明 KD support 已删除
- **AND** 错误信息 MUST 指向对应 supervised、strong 或 lightweight 配置入口

#### Scenario: 旧 distillation override 被拒绝
- **WHEN** 用户传入命令行覆盖 `distillation.type=logits_kd`、`distillation.type=rkd`、`distillation.teacher_model_name=...` 或任意 `distillation.*` 字段
- **THEN** 配置解析 MUST 失败
- **AND** 系统 MUST 不静默忽略该字段或回退为 supervised 配置
### Requirement: 历史产物只读保留
删除 KD 支持 MUST NOT 自动删除本地历史运行产物、日志、checkpoint 或 OpenSpec archive。历史 artifact 中已有 KD 字段 MAY 被只读工具展示，但不得成为新运行契约。

#### Scenario: 实现变更不清理 outputs
- **WHEN** 开发者实现本 change
- **THEN** 变更 MUST 不自动删除 `outputs/`、`logs/`、`All_models/`、dataset、cache 或历史 checkpoint
- **AND** 如需清理本地产物，系统 MUST 使用 runtime cleanup manifest 流程
