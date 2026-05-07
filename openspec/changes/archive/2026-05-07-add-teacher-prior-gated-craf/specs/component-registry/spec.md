## ADDED Requirements

### Requirement: Teacher-prior CRAF 组件注册
项目 MUST 通过现有组件注册和默认组件导入边界暴露 teacher-prior CRAF 所需组件。新增模型、gate、loss、KD loss 和 helper MUST 可由配置或窄模块导入复用，并且不得要求复制训练脚本。

#### Scenario: 注册 PriorResidualGate 或 gate factory
- **WHEN** 配置选择 `gate_type: prior_residual_sigmoid`
- **THEN** 系统 MUST 能构建 prior residual gate
- **AND** 构建失败时错误信息 MUST 包含 gate 类型和可用 gate 类型

#### Scenario: 注册 teacher-prior CRAF 模型入口
- **WHEN** 配置选择 teacher-prior CRAF 所需模型类型
- **THEN** `MODELS` 注册表 MUST 能构建对应模型
- **AND** `import_default_components()` 后可用模型列表 MUST 包含该模型或继续包含可承载该 gate 的 `craf_fusion`

#### Scenario: 注册 prior 和 KD loss
- **WHEN** 配置显式启用 prior regularization 或 reliability-weighted KD
- **THEN** 系统 MUST 能通过现有 loss/distillation 构建边界调用对应 loss
- **AND** 关闭这些 loss 时训练流程 MUST 不构建无用组件

### Requirement: Teacher loader 组件边界
teacher encoder loader MUST 以窄模块函数或可测试组件提供。loader MUST 不依赖训练循环内部局部变量，并 MUST 能在单元测试中用合成 checkpoint 验证 key mapping、strict 模式和冻结策略。

#### Scenario: 单元测试直接调用 teacher loader
- **WHEN** 测试用合成 teacher checkpoint 调用 teacher loader
- **THEN** loader MUST 返回每模态 load summary
- **AND** loader MUST 能在没有 dataloader 或 trainer 的情况下运行

#### Scenario: strict 模式抛出清晰错误
- **WHEN** strict loader 遇到 shape mismatch
- **THEN** loader MUST 抛出包含模态、checkpoint 路径和 mismatch key 的错误

### Requirement: 默认导入保持轻量
新增 teacher-prior CRAF 组件 MUST 遵守现有轻量导入约束。导入 `kd_sensing.registries` MUST 不急切导入训练器、dataset 或 checkpoint 文件；默认组件导入 MUST 仍由构建流程显式触发。

#### Scenario: 轻量导入 registry 不触发训练模块
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 teacher registry 构建脚本或 trainer 模块

#### Scenario: 构建 CRAF 前导入默认组件
- **WHEN** 构建流程调用 `import_default_components()` 后再查询 `MODELS`
- **THEN** teacher-prior CRAF 相关内置模型或 gate 所在模块 MUST 已完成注册
