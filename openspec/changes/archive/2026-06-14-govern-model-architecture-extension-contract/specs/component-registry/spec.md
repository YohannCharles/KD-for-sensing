## ADDED Requirements

### Requirement: 新整模型注册受治理
组件注册系统 MUST 继续支持 `MODELS` 注册整模型，但新增整模型注册 MUST 被视为架构例外并纳入 OpenSpec、文档和测试护栏。新增普通 baseline MUST 优先注册 encoder/projector/representation core/head 子组件，而不是注册新的整模型。

#### Scenario: 子组件注册优先
- **WHEN** 新增模型能力可以表达为 encoder、projector、representation core 或 head
- **THEN** 实现 MUST 使用对应子组件 registry
- **AND** 不得仅为组合这些子组件而新增新的 `MODELS` 注册名

#### Scenario: 整模型注册需要例外说明
- **WHEN** 新增源码包含新的 `@MODELS.register(...)` 或等价模型注册
- **THEN** 对应 change MUST 提供 whole-model exception 理由
- **AND** focused tests MUST 覆盖 registry build、forward 输出、metadata 和轻量导入边界

### Requirement: 默认组件导入登记新增模型组件
新增内置模型子组件或整模型例外 MUST 被默认组件导入流程显式登记，同时保持 `kd_sensing.registries` 轻量可导入。默认组件导入 MUST 不通过兼容 facade、仓库扫描或旧聚合模块发现组件。

#### Scenario: 新组件可通过默认导入发现
- **WHEN** 构建流程调用 `import_default_components()` 后查询对应 registry
- **THEN** 新增内置 encoder/projector/core/head 或例外模型注册名 MUST 出现在 registry 列表中
- **AND** 仅导入 `kd_sensing.registries` MUST 不 eager import dataset、trainer、torchvision 权重接口或 checkpoint 文件

### Requirement: 扩展文档区分默认和例外注册
组件发现和扩展文档 MUST 将新增 baseline 的默认路径描述为模块化配置或子组件注册。直接注册 `MODELS` 的示例 MUST 位于 whole-model exception 小节，并说明需要 OpenSpec 设计理由和 focused tests。

#### Scenario: 文档默认示例使用模块化组件
- **WHEN** 开发者阅读 Add a Model 或新增 baseline 指南
- **THEN** 首个示例 MUST 展示 `modular_sequence` 配置或子组件 registry
- **AND** 文档 MUST 不把直接 `@MODELS.register` 整模型作为普通 baseline 的默认建议
