## MODIFIED Requirements

### Requirement: 组件注册表
项目 MUST 提供轻量组件注册表，用于注册和构建模型、数据集、损失函数、指标和预处理器。注册表 MUST 支持按字符串名称查询组件，并通过配置参数实例化组件。项目 MUST 不再提供 `DISTILLERS` registry 或默认 distiller 注册流程。

#### Scenario: 按名称构建模型
- **WHEN** 配置中指定一个已注册模型名称和初始化参数
- **THEN** 系统 MUST 返回对应模型实例
- **AND** 系统 MUST 将配置参数传入模型构造函数

#### Scenario: registry 不暴露 distillers
- **WHEN** 开发者导入 `kd_sensing.registries`
- **THEN** 模块 MUST 不导出 `DISTILLERS`
- **AND** 默认组件导入 MUST 不导入 `kd_sensing.distillation.distillers`

## REMOVED Requirements

### Requirement: 可扩展蒸馏方法
**Reason**: 项目删除 teacher-student KD 支持，distiller 扩展点不再属于受支持架构。
**Migration**: 新监督损失放入 loss/objective/extension 模块；未来蒸馏方法必须通过新的 OpenSpec change 重新定义。

#### Scenario: 选择 logits KD
- **WHEN** 配置中选择 logits KD
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指向 supervised 或 adaptation 入口

#### Scenario: 选择 relational KD
- **WHEN** 配置中选择 relational KD
- **THEN** 系统 MUST 拒绝配置
- **AND** 系统 MUST 不构建 distiller

