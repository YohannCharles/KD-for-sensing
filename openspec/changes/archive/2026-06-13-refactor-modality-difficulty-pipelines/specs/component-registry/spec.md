## ADDED Requirements

### Requirement: Difficulty operator 注册表
项目 MUST 提供 difficulty operator 注册边界，用于按字符串名称注册、查询和构建 GPS、image 和未来模态输入难度 operator。该注册边界 MAY 复用现有 `Registry` 实现或新增窄 registry，但 MUST 保持轻量导入，不得在导入 registry 时 eager import dataset、model、diagnostics renderer、training loop 或大型视觉依赖。

#### Scenario: 按名称构建 GPS delay operator
- **WHEN** 配置指定 difficulty operator `gps_temporal_delay` 及其参数
- **THEN** 系统 MUST 通过 difficulty operator registry 构建该 operator
- **AND** 训练、评估和 benchmark MUST 能复用同一注册名

#### Scenario: 轻量导入 difficulty registry
- **WHEN** 开发者执行 `import kd_sensing.registries` 或导入 difficulty registry 窄模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入默认 dataset、model、diagnostics renderer、torchvision 权重接口或训练循环

#### Scenario: 未知 difficulty operator 错误可诊断
- **WHEN** 配置引用未注册 difficulty operator
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含 registry 名称、请求 operator 和可用 operator 列表

### Requirement: 默认 difficulty operator 显式注册
内置 difficulty operators MUST 通过显式默认组件导入或 difficulty 专用默认注册函数完成注册。构建流程在解析或应用 difficulty profile 前 MUST 触发该注册动作；仅导入 registry 对象 MUST 不自动注册所有重依赖 operator。

#### Scenario: 构建前导入默认 difficulty operators
- **WHEN** 配置加载或 benchmark runner 需要解析内置 GPS/image difficulty profile
- **THEN** 构建流程 MUST 先触发默认 difficulty operator 注册
- **AND** registry MUST 包含 GPS noise、GPS async、image degradation 等内置注册名

#### Scenario: 自定义 difficulty operator 可插拔
- **WHEN** 开发者在自定义模块中注册新的 image difficulty operator 并在配置中引用
- **THEN** 系统 MUST 能在该模块被显式导入后解析并构建该 operator
- **AND** 训练和 benchmark 主循环 MUST 不需要为该 operator 增加专用分支
