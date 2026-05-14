## ADDED Requirements

### Requirement: CLS-token Transformer fusion 组件注册
项目 MUST 通过现有组件注册表暴露 CLS-token Transformer fusion 模型。新增模型 MUST 能通过 `MODELS` 注册表构建，并 MUST 复用现有 fusion 训练、验证和评估入口。

#### Scenario: 按名称构建 CLS-token Transformer fusion
- **WHEN** 配置指定 `type: cls_token_transformer_fusion`
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 CLS-token Transformer fusion 模型实例
- **AND** 构建参数 MUST 来自配置字段
- **AND** 模型 MUST 支持现有 fusion forward 输入键

#### Scenario: 注册错误可诊断
- **WHEN** 用户引用不存在或拼写错误的 CLS-token Transformer fusion 注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称和可用模型注册名

### Requirement: 默认组件导入包含 CLS-token Transformer fusion
默认组件导入流程 MUST 注册 CLS-token Transformer fusion 内置模型，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 dataset、trainer、checkpoint 或重依赖运行模块。

#### Scenario: 构建流程导入默认组件
- **WHEN** 构建流程调用 `import_default_components()` 后再查询 `MODELS`
- **THEN** `MODELS` 注册表 MUST 包含 `cls_token_transformer_fusion`
- **AND** 系统 MUST 能通过配置构建该模型

#### Scenario: 轻量导入 registry
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import CLS-token Transformer fusion 模型依赖

#### Scenario: 内置组件列表可发现
- **WHEN** 开发者按扩展文档触发默认模型模块导入后查看 `MODELS.list()`
- **THEN** 输出 MUST 包含 `cls_token_transformer_fusion`
- **AND** 输出 MUST 继续包含现有 canonical fusion 模型注册名
