## ADDED Requirements

### Requirement: CRAF 组件注册
CRAF 相关模型和 loss 组件 MUST 通过现有组件注册或明确的窄模块入口接入系统。新增组件 MUST 能通过配置名称构建，并 MUST 不要求训练脚本手写实例化逻辑。

#### Scenario: 按名称构建 CRAF 模型
- **WHEN** 配置指定 `type: craf_fusion`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 CRAF 模型
- **AND** 构建参数 MUST 来自配置字段

#### Scenario: 按名称构建 token transformer baseline
- **WHEN** 配置指定 token transformer fusion baseline 的注册名
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建该 baseline

#### Scenario: 注册错误可诊断
- **WHEN** 用户引用不存在的 CRAF 组件注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称和可用组件列表

### Requirement: 默认组件导入包含 CRAF
默认组件导入流程 MUST 注册 CRAF 内置组件，同时保持 registry 轻量导入边界。

#### Scenario: 构建流程导入默认组件
- **WHEN** 构建流程调用默认组件导入函数后再构建 `craf_fusion`
- **THEN** `MODELS` 注册表 MUST 包含 CRAF 注册名

#### Scenario: 轻量导入 registry
- **WHEN** 开发者仅导入 `kd_sensing.registries`
- **THEN** 系统 MUST 不 eager import CRAF 模型依赖
- **AND** 轻量导入边界 MUST 与现有 registry 语义一致

### Requirement: CRAF loss helper 可测试
CRAF 使用的 beam soft loss、sequence CE/per-sample loss 和 gate supervision helper MUST 有明确模块边界，并 MUST 能被单元测试直接调用。

#### Scenario: 直接测试 beam soft loss
- **WHEN** 测试代码传入 logits、labels、beam 数量和 sigma
- **THEN** helper MUST 返回标量 loss
- **AND** ignore index 位置 MUST 不影响 loss

#### Scenario: 直接测试 gate target
- **WHEN** 测试代码传入 full loss 与 drop loss
- **THEN** helper MUST 返回范围可控的模态贡献目标
- **AND** 目标 MUST 能与 reliability gate 计算监督 loss
